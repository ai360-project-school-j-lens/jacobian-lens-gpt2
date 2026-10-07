"""Readout-position cache for fast lens comparisons on the evaluation sets.

The dataset protocol (:mod:`jlens.evaluation`) ranks words at a single readout
position per prompt. Caching the pre-final-norm block outputs there,
``H[item, layer]``, turns any readout of those residuals -- the logit lens, a
J-lens, a lens averaged over any subset of fit prompts, block-output deltas --
into a vocabulary readout plus ranking, with no further model forward passes.

Words and ranks follow ``jlens.evaluation._readout_rows`` exactly, so
:func:`words_frame` output works with :func:`jlens.evaluation.pass_at_k`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jlens.evaluation import readout_position, single_token_ids, spelling_ids
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclasses.dataclass
class ReadoutCache:
    """Readout-position residuals and the words ranked there.

    Attributes:
        H: ``[n_items, n_layers, d_model]`` fp32 block outputs (pre-final-norm);
            the last layer is the model's final block.
        items: A row per eval item (``dataset``, ``item``, ``target``).
        words: A row per ranked word, as in ``_readout_rows``, plus
            ``item_index`` (row of ``items``/``H``) and ``ids`` (spelling ids).
    """

    H: torch.Tensor
    items: pd.DataFrame
    words: pd.DataFrame

    def save(self, path: str | Path) -> None:
        torch.save({"H": self.H, "items": self.items.to_dict("list"),
                    "words": self.words.to_dict("list")}, path)

    @classmethod
    def load(cls, path: str | Path) -> ReadoutCache:
        raw = torch.load(path, weights_only=False)
        return cls(raw["H"], pd.DataFrame(raw["items"]), pd.DataFrame(raw["words"]))


def _special_ids(tokenizer) -> set[int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    special.update(
        token for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (token := getattr(tokenizer, attr, None)) is not None
    )
    return special


@torch.no_grad()
def build_readout_cache(model: LensModel, evals: dict[str, list[dict]]) -> ReadoutCache:
    """One forward per item; keep every block output at the readout position."""
    tok = model.tokenizer
    special = _special_ids(tok)
    residuals, item_rows, word_rows = [], [], []
    for dataset, samples in evals.items():
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
            words += [("control", word, role) for role, word in enumerate(other)
                      if word not in item["intermediates"]]
            if "target" in item:
                words.append(("target", item["target"], 0))
            prompt_ids = {token for token in ids if token not in special}
            for kind, word, role in words:
                word_ids = spelling_ids(tok, word, expand)
                word_rows.append({
                    "item_index": item_index, "dataset": dataset, "item": item["name"],
                    "kind": kind, "word": word, "role": role,
                    "single_token": bool(single_token_ids(tok, word, expand)),
                    "in_prompt": bool(word_ids & prompt_ids),
                    "ids": sorted(word_ids),
                })
            del recorder
    return ReadoutCache(torch.stack(residuals), pd.DataFrame(item_rows), pd.DataFrame(word_rows))


class WordRanker:
    """Ranks of the cached words in vocabulary logits at the readout positions."""

    def __init__(self, words: pd.DataFrame, device: str | torch.device = "cpu") -> None:
        self.n_words = len(words)
        width = int(words.ids.map(len).max())
        ids = torch.zeros(self.n_words, width, dtype=torch.long)
        mask = torch.zeros(self.n_words, width, dtype=torch.bool)
        for row, word_ids in enumerate(words.ids):
            ids[row, : len(word_ids)] = torch.tensor(word_ids)
            mask[row, : len(word_ids)] = True
        self.device = torch.device(device)
        self.ids, self.mask = ids.to(self.device), mask.to(self.device)
        self.word_item = torch.tensor(words.item_index.values, device=self.device)

    @torch.no_grad()
    def __call__(self, logits: torch.Tensor, row_chunk: int = 512) -> np.ndarray:
        """``logits [..., n_items, vocab]`` -> ranks ``[..., n_words]``.

        1-based rank among all tokens, minimum over the word's spellings (ties
        resolved as in :func:`jlens.evaluation.token_ranks`).
        """
        lead, (n_items, vocab) = logits.shape[:-2], logits.shape[-2:]
        flat = logits.reshape(-1, vocab).to(self.device)
        batch = flat.shape[0] // n_items
        rows = (torch.arange(batch, device=self.device)[:, None] * n_items + self.word_item[None]).reshape(-1)
        ids, mask = self.ids.repeat(batch, 1), self.mask.repeat(batch, 1)
        ranks = torch.empty(len(rows), dtype=torch.int32, device=self.device)
        for start in range(0, len(rows), row_chunk):
            chunk = slice(start, start + row_chunk)
            word_logits = flat[rows[chunk]].float()
            best = word_logits.gather(1, ids[chunk]).masked_fill(~mask[chunk], -torch.inf).max(1).values
            ranks[chunk] = (word_logits > best[:, None]).sum(1).int() + 1
        return ranks.reshape(*lead, self.n_words).cpu().numpy()

    @torch.no_grad()
    def rank_residuals(
        self,
        readout: Callable[[torch.Tensor], torch.Tensor],
        residuals: torch.Tensor,
        batch: int = 4,
    ) -> np.ndarray:
        """``readout`` (e.g. ``model.unembed``) of ``residuals [B, n_items, d]`` -> ranks ``[B, n_words]``.

        ``batch`` bounds memory: ``batch * n_items`` vocabulary rows at a time.
        """
        out = []
        for start in range(0, residuals.shape[0], batch):
            x = residuals[start:start + batch].to(self.device).float()
            out.append(self(readout(x)))
        return np.concatenate(out)


def words_frame(words: pd.DataFrame, ranks: np.ndarray) -> pd.DataFrame:
    """``_readout_rows``-style frame from ranks ``[n_words, n_layers]`` (last = model output)."""
    frame = words.drop(columns=["ids", "item_index"]).copy()
    frame["ranks"] = list(ranks)
    frame["best_rank"] = ranks.min(1)
    frame["best_layer"] = ranks.argmin(1)
    frame["inner_best"] = ranks[:, :-1].min(1)
    return frame


def item_scores(
    words: pd.DataFrame,
    ranks: np.ndarray,
    dataset: str,
    kind: str,
    k: int | None = 10,
    layers: slice | Sequence[int] = slice(None, -1),
) -> np.ndarray:
    """Per-item score of one dataset and word kind; batched over leading axes.

    ``ranks [..., n_words, n_layers]``. A word's score is ``best rank <= k``
    over ``layers`` (default: inner layers), or ``log10`` of that best rank if
    ``k`` is None; an item's score is the mean over its words of this kind.
    Returns ``[..., n_items_of_dataset]`` in ``item_index`` order, NaN for items
    without such words. The mean over items of ``k`` scores is pass@k.
    """
    in_dataset = (words.dataset == dataset).values
    items = np.unique(words.item_index.values[in_dataset])
    sel = in_dataset & (words.kind == kind).values
    best = ranks[..., sel, :][..., layers].min(-1)
    x = np.log10(best) if k is None else (best <= k).astype(float)
    membership = np.zeros((len(items), int(sel.sum())))
    membership[np.searchsorted(items, words.item_index.values[sel]), np.arange(int(sel.sum()))] = 1
    counts = membership.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, (x @ membership.T) / counts, np.nan)
