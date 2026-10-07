"""Shared pieces for J-lens convergence / confidence-interval experiments.

The J-lens is linear in J: ``J_S h = mean_{p in S} J_p h``. So for every fit
prompt ``p`` we store ``T_p[l] = H_l @ J_p[l].T`` -- the eval-set residuals at the
readout positions transported by that prompt's own Jacobian. A lens fitted on
any subset S of prompts is then read out on the eval set as ``mean_{p in S} T_p``
without refitting: only ``unembed`` and ranks remain (about a second on GPU).

Ranks and words follow ``jlens.evaluation._readout_rows`` exactly, so the
resulting ``words`` frame plugs into ``jlens.evaluation.pass_at_k``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gpt2 import GPT2LensModel
from jlens.evaluation import (
    DATASETS,
    FIT_MIX,
    load_eval,
    load_fit_prompts,
    readout_position,
    single_token_ids,
    spelling_ids,
)
from jlens.hooks import ActivationRecorder

MODEL_ID = "openai-community/gpt2-xl"
REPO_DIR = Path(__file__).resolve().parents[2]
SNAPSHOT_FRACTION = 0.25  # the 134-prompt snapshot lens: FIT_MIX * 0.25, seed 0


def load_model(device: str = "cuda") -> GPT2LensModel:
    return GPT2LensModel.from_pretrained(MODEL_ID, device=device)


def fit_mix(fraction: float) -> dict[str, int]:
    """Same rounding as the dataset notebooks."""
    return {source: max(1, round(n * fraction)) for source, n in FIT_MIX.items()}


def build_prompt_list(path: Path, extended_fraction: float = 1.0) -> pd.DataFrame:
    """Snapshot prompts in their original order, then the extension prompts.

    ``load_fit_prompts`` takes the first n qualifying records of every source,
    so the snapshot corpus is a per-source prefix of any larger fraction.
    The seed only shuffles the order; it never changes the corpus.
    """
    if path.exists():
        return pd.read_json(path, orient="records", lines=True)
    snapshot = load_fit_prompts(fit_mix(SNAPSHOT_FRACTION), seed=0).assign(part="snapshot")
    full = load_fit_prompts(fit_mix(extended_fraction), seed=0)
    extension = full[~full.text.isin(set(snapshot.text))].assign(part="extension")
    missing = set(snapshot.text) - set(full.text)
    if missing:
        raise RuntimeError(f"{len(missing)} snapshot prompts are not a prefix of the larger corpus")
    prompts = pd.concat([snapshot, extension], ignore_index=True)
    prompts.to_json(path, orient="records", lines=True, force_ascii=False)
    return prompts


@torch.no_grad()
def build_eval_cache(model: GPT2LensModel) -> dict:
    """Readout-position residuals of every eval item and the words to rank there.

    Returns ``H`` [n_items, n_layers, d_model] fp32 (pre-``ln_f`` block outputs,
    layer ``n_layers - 1`` is the final one), an ``items`` frame and a ``words``
    frame mirroring ``_readout_rows`` (``ids`` holds each word's spelling ids).
    """
    tok = model.tokenizer
    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    special_ids.update(
        token for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (token := getattr(tok, attr, None)) is not None
    )
    residuals, item_rows, word_rows = [], [], []
    for dataset in DATASETS:
        samples = load_eval(str(REPO_DIR), dataset)
        expand = dataset == "order-ops"
        for i, item in enumerate(samples):
            input_ids = model.encode(item["prompt"].rstrip())
            ids = input_ids[0].tolist()
            with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
                model.forward(input_ids)
            position = readout_position(tok, ids, dataset)
            residuals.append(torch.stack([
                recorder.activations[layer][0, position].float().cpu()
                for layer in range(model.n_layers)
            ]))
            item_index = len(item_rows)
            item_rows.append({"dataset": dataset, "item": item["name"], "target": item.get("target")})

            other = samples[(i + len(samples) // 2) % len(samples)]["intermediates"]
            words = [("intermediate", word, role) for role, word in enumerate(item["intermediates"])]
            words += [("control", word, role) for role, word in enumerate(other) if word not in item["intermediates"]]
            if "target" in item:
                words.append(("target", item["target"], 0))
            prompt_ids = {token for token in ids if token not in special_ids}
            for kind, word, role in words:
                word_ids = spelling_ids(tok, word, expand)
                word_rows.append({
                    "item_index": item_index,
                    "dataset": dataset,
                    "item": item["name"],
                    "kind": kind,
                    "word": word,
                    "role": role,
                    "single_token": bool(single_token_ids(tok, word, expand)),
                    "in_prompt": bool(word_ids & prompt_ids),
                    "ids": sorted(word_ids),
                })
            del recorder
    return {
        "H": torch.stack(residuals),
        "items": pd.DataFrame(item_rows),
        "words": pd.DataFrame(word_rows),
    }


def save_eval_cache(cache: dict, path: Path) -> None:
    torch.save({
        "H": cache["H"],
        "items": cache["items"].to_dict("list"),
        "words": cache["words"].to_dict("list"),
    }, path)


def load_eval_cache(path: Path) -> dict:
    raw = torch.load(path, weights_only=False)
    return {"H": raw["H"], "items": pd.DataFrame(raw["items"]), "words": pd.DataFrame(raw["words"])}


class Ranker:
    """Ranks of the cached words for readout residuals of any lens."""

    def __init__(self, model: GPT2LensModel, cache: dict, device: str = "cuda") -> None:
        self.model = model
        self.words = cache["words"]
        self.device = device
        n_ids = self.words.ids.map(len).max()
        ids = torch.zeros(len(self.words), n_ids, dtype=torch.long)
        mask = torch.zeros(len(self.words), n_ids, dtype=torch.bool)
        for row, word_ids in enumerate(self.words.ids):
            ids[row, : len(word_ids)] = torch.tensor(word_ids)
            mask[row, : len(word_ids)] = True
        self.ids, self.mask = ids.to(device), mask.to(device)
        self.word_item = torch.tensor(self.words.item_index.values, device=device)
        self.final = self.layer_ranks(cache["H"][:, -1])

    @torch.no_grad()
    def layer_ranks(self, residuals: torch.Tensor) -> np.ndarray:
        """Ranks [n_words] for one layer's readout residuals [n_items, d_model]."""
        logits = self.model.unembed(residuals.to(self.device).float()).float()  # [n_items, vocab]
        word_logits = logits[self.word_item]  # [n_words, vocab]
        best = word_logits.gather(1, self.ids).masked_fill(~self.mask, -torch.inf).max(1).values
        return ((word_logits > best[:, None]).sum(1) + 1).cpu().numpy()

    def words_frame(self, transported: torch.Tensor) -> pd.DataFrame:
        """``words`` frame for inner-layer residuals [n_inner, n_items, d_model] + the model output."""
        ranks = np.stack([self.layer_ranks(transported[layer]) for layer in range(transported.shape[0])]
                         + [self.final], axis=1)
        frame = self.words.drop(columns=["ids", "item_index"]).copy()
        frame["ranks"] = list(ranks)
        frame["best_rank"] = ranks.min(1)
        frame["best_layer"] = ranks.argmin(1)
        frame["inner_best"] = ranks[:, :-1].min(1)
        return frame


def transport_with(jacobians: dict[int, torch.Tensor], H: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """[n_inner, n_items, d_model]: ``H[:, l] @ J_l.T`` for every inner layer."""
    return torch.stack([
        H[:, layer].to(device) @ jacobians[layer].to(device, torch.float32).T
        for layer in sorted(jacobians)
    ])


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
