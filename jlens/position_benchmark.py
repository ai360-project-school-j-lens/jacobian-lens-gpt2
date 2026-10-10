"""Batched full-position ranks and tokenizer-independent latent-bridge spans."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from jlens.batched_evaluation import (
    _batch_mode,
    _batches,
    _decode,
    _forward_batch,
    _prepare,
)
from jlens.lens import JacobianLens
from jlens.readout import selected_token_ranks
from jlens.strict_scoring import ExplicitPromptModel


def load_latent_bridge(repo_dir: str | Path) -> dict[str, list[dict]]:
    """Load all candidates, retaining the original GPT-2 selection as metadata."""
    root = Path(repo_dir) / "data" / "experiments"
    pool = json.loads((root / "latent-bridge-v2-candidates.json").read_text())
    kept = json.loads((root / "latent-bridge-v2.json").read_text())
    kept = {(item["family"], item["subject"]) for item in kept["items"]}
    result = {}
    for family, spec in pool["families"].items():
        items = []
        for country, subjects in spec["subjects"].items():
            for subject in subjects:
                prompt = spec["template"].format(X=subject)
                prompt = prompt[0].upper() + prompt[1:]
                items.append({
                    "name": subject, "prompt": prompt, "subject": subject,
                    "family": family, "intermediates": [country],
                    "target": pool["languages"][country],
                    "gpt2_kept": (family, subject) in kept,
                })
        # Keep controls from the same family and guarantee a different country.
        for i, item in enumerate(items):
            for shift in range(len(items)):
                other = items[(i + len(items) // 2 + shift) % len(items)]
                if other["intermediates"] != item["intermediates"]:
                    item["controls"] = other["intermediates"]
                    break
            else:
                raise ValueError(f"No distinct control country for {family}")
        result[f"latent-bridge/{family}"] = items
    return result


def latent_bridge_spans(
    model: ExplicitPromptModel, item: dict, *, max_seq_len: int = 512,
) -> dict[str, list[int]]:
    """Map character spans to real token positions using fast-tokenizer offsets.

    The slot is the final token of the preposition before ``a country``.
    Country includes every fragment of ``country``. A token crossing a span
    boundary is included by overlap. No GPT-2 token strings are assumed.
    """
    if not getattr(model.tokenizer, "is_fast", False):
        raise ValueError("Latent-bridge spans require a fast tokenizer with offsets")
    prompt = item["prompt"]
    encoded = model.tokenizer(
        prompt, add_special_tokens=model.bos_policy == "tokenizer",
        return_offsets_mapping=True,
    )
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if model.bos_policy == "prepend":
        ids.insert(0, model.tokenizer.bos_token_id)
        offsets.insert(0, (0, 0))
    if ids != model.encode_ids(prompt, max_length=max_seq_len):
        raise ValueError("Offset encoding and evaluation encoding disagree")
    special = set(getattr(model.tokenizer, "all_special_ids", []) or [])
    special.update(t for t in (
        model.tokenizer.bos_token_id, model.tokenizer.eos_token_id,
        model.tokenizer.pad_token_id,
    ) if t is not None)
    valid = [i for i, (a, b) in enumerate(offsets) if b > a and ids[i] not in special]

    def overlap(start, end):
        return [i for i in valid if offsets[i][0] < end and offsets[i][1] > start]

    match = re.search(r"\b(in|from) a (country)\b", prompt)
    if match is None or not prompt.lower().startswith(item["subject"].lower()):
        raise ValueError("Unrecognized latent-bridge template")
    subject = overlap(0, len(item["subject"]))
    preposition = overlap(*match.span(1))
    country = overlap(*match.span(2))
    if not subject or not preposition or not country or not valid:
        raise ValueError("Tokenizer produced an empty required span")
    slot, last = [preposition[-1]], [valid[-1]]
    return {
        "subject": subject, "country token": country,
        "other template": sorted(set(valid) - set(subject + slot + country + last)),
        "last": last, "all but slot": sorted(set(valid) - set(slot)),
        "slot": slot, "all": valid,
    }


@torch.inference_mode()
def evaluate_position_ranks(
    model: ExplicitPromptModel,
    lens: JacobianLens,
    evals: dict[str, list[dict]],
    *,
    spelling_lookup: Callable,
    batch_size: int = 8,
    max_batch_tokens: int = 2048,
    max_seq_len: int = 512,
    position_chunk_size: int = 8,
    rank_chunk_size: int = 8,
    batching: str = "auto",
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank whole-token words at every real non-special prompt position.

    Returns word rows with ``ranks[layer, position]`` and absolute ``positions``,
    plus one item row with final-token answer diagnostics. Empty accepted-ID
    sets retain null ranks. Explicit ``controls`` override the standard shifted
    item controls. Matrices are moved once per layer/device; model passes share
    the batching, mask and readout implementation of ``batched_evaluation``.
    Vocabulary workspace is bounded by the two chunk sizes; block activations
    for one batch and all inner Jacobians remain resident. CPU/CUDA only.
    """
    if not isinstance(model, ExplicitPromptModel):
        raise TypeError("Supply ExplicitPromptModel with an explicit BOS policy")
    for value in (batch_size, max_batch_tokens, max_seq_len,
                  position_chunk_size, rank_chunk_size):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("Batch, length and chunk limits must be positive integers")
    if model.n_layers < 2 or lens.d_model != model.d_model:
        raise ValueError("Need at least two blocks and a matching lens width")
    for layer in range(model.n_layers - 1):
        if layer not in lens.jacobians or lens.jacobians[layer].shape != (
            model.d_model, model.d_model,
        ):
            raise ValueError(f"Missing or incompatible Jacobian at layer {layer}")
    padded = _batch_mode(model, batching)
    samples = _prepare(model, evals, spelling_lookup, max_seq_len)
    if not samples:
        raise ValueError("Supply at least one evaluation item")
    for sample in samples:
        if "controls" in sample.item:
            sample.words = [w for w in sample.words if w["kind"] != "control"]
            for role, word in enumerate(sample.item["controls"]):
                accepted = tuple(sorted(spelling_lookup(word, sample.dataset == "order-ops")))
                special = set(getattr(model.tokenizer, "all_special_ids", []) or [])
                special.update(
                    token for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
                    if (token := getattr(model.tokenizer, attr, None)) is not None
                )
                if any(not isinstance(t, int) or isinstance(t, bool) or t < 0
                       or t in special for t in accepted):
                    raise ValueError("Accepted IDs must be non-special token IDs")
                sample.words.append({
                    "dataset": sample.dataset, "item": sample.item["name"],
                    "kind": "control", "word": word, "role": role,
                    "single_token": bool(accepted), "accepted_ids": accepted,
                    "in_prompt": bool(set(accepted).intersection(sample.ids)),
                })
    batches = list(_batches(samples, batch_size, max_batch_tokens, padded))
    matrices, results = {}, {}
    layers = model.n_layers
    for batch in tqdm(batches, disable=not progress, desc="all-position ranks"):
        activations = _forward_batch(model, batch, padded)
        coordinates = [(b, p) for b, s in enumerate(batch) for p in s.valid]
        width = max((len(w["accepted_ids"]) for s in batch for w in s.words), default=1)
        width = max(width, 1)
        word_width = max(len(s.words) for s in batch)
        accepted = torch.zeros(len(batch), word_width, width, dtype=torch.long)
        supported = torch.zeros(len(batch), word_width, dtype=torch.bool)
        for b, sample in enumerate(batch):
            for w, word in enumerate(sample.words):
                ids = word["accepted_ids"]
                if ids:
                    accepted[b, w] = torch.tensor([*ids, *([ids[0]] * (width - len(ids)))])
                    supported[b, w] = True
        # Resolve output placement once, including models with a sharded head.
        first = activations[layers - 1]
        probe = _decode(model, lens, activations, layers - 1,
                        torch.tensor([0], device=first.device),
                        torch.tensor([batch[0].valid[-1]], device=first.device),
                        False, matrices)
        device, vocab = probe.logits.device, probe.logits.shape[-1]
        del probe
        if accepted.min() < 0 or accepted.max() >= vocab:
            raise ValueError("Accepted token ID outside the output vocabulary")
        accepted, supported = accepted.to(device), supported.to(device)
        ranks = torch.empty(2, layers, len(coordinates), word_width,
                            dtype=torch.int32, device=device)
        top1 = torch.empty(len(coordinates), dtype=torch.long, device=device)
        finite = torch.ones((), dtype=torch.bool, device=device)
        for start in range(0, len(coordinates), position_chunk_size):
            chunk = coordinates[start:start + position_chunk_size]
            indices = {
                activation_device: (
                    torch.tensor([b for b, _ in chunk], device=activation_device),
                    torch.tensor([p for _, p in chunk], device=activation_device),
                )
                for activation_device in {h.device for h in activations.values()}
            }
            owners = torch.tensor([b for b, _ in chunk], device=device)
            token_ids = accepted[owners].reshape(len(chunk), -1)
            for layer in range(layers):
                for li in range(1 if layer == layers - 1 else 2):
                    rows, positions = indices[activations[layer].device]
                    result = _decode(model, lens, activations, layer, rows, positions,
                                     li == 1, matrices)
                    scores = result.ranking_scores
                    finite &= torch.isfinite(scores).all() & torch.isfinite(result.logits).all()
                    ranked = []
                    for offset in range(0, len(chunk), rank_chunk_size):
                        sl = slice(offset, offset + rank_chunk_size)
                        ranked.append(selected_token_ranks(scores[sl], token_ids[sl]))
                    values = torch.cat(ranked).reshape(len(chunk), word_width, width).min(-1).values
                    ranks[li, layer, start:start + len(chunk)] = values.masked_fill(~supported[owners], -1)
                    if layer == layers - 1:
                        top1[start:start + len(chunk)] = result.logits.argmax(-1)
                    del result, scores
            ranks[1, -1, start:start + len(chunk)] = ranks[0, -1, start:start + len(chunk)]
        if not bool(finite.cpu()):
            raise ValueError("Nonfinite readout scores")
        ranks, top1 = ranks.cpu().numpy(), top1.cpu().numpy()
        offset = 0
        for sample in batch:
            stop = offset + len(sample.valid)
            words = []
            for li, name in enumerate(("logit lens", "J-lens")):
                for w, word in enumerate(sample.words):
                    words.append({**word, "lens": name, "positions": sample.valid,
                                  "ranks": ranks[li, :, offset:stop, w].copy()
                                  if word["single_token"] else None})
            prediction = int(top1[stop - 1])
            item = {
                **sample.item, "dataset": sample.dataset, "item": sample.item["name"],
                "ids": sample.ids, "model_top1_id": prediction,
                "model_top1": model.tokenizer.decode([prediction]),
                "model_correct": prediction in sample.target_ids if sample.target_ids else None,
            }
            results[sample.index] = words, item
            offset = stop
        del activations
    words = pd.DataFrame([w for i in range(len(samples)) for w in results[i][0]])
    items = pd.DataFrame([results[i][1] for i in range(len(samples))])
    for frame in (words, items):
        frame.attrs["evaluation"] = {
            "n_model_passes": len(batches), "batch_sizes": [len(b) for b in batches],
            "batching": "padded" if padded else "equal_length",
            "bos_policy": model.bos_policy,
        }
    return words, items


def reduce_position_ranks(words: pd.DataFrame, spans: dict) -> pd.DataFrame:
    """Min-reduce positions within each span into ``pass_at_k`` word rows.

    ``spans[(dataset, item)][span_name]`` contains absolute token positions.
    Empty spans and unsupported words retain null ranks for coverage reporting.
    """
    rows = []
    for word in words.to_dict("records"):
        mapping = {p: i for i, p in enumerate(word["positions"])}
        for name, positions in spans[word["dataset"], word["item"]].items():
            if set(positions) - mapping.keys():
                raise ValueError("Span includes a special or absent token position")
            columns = [mapping[p] for p in positions]
            rank = (word["ranks"][:, columns].min(1)
                    if word["ranks"] is not None and columns else None)
            rows.append({**word, "span": name, "ranks": rank,
                         "n_positions": len(columns),
                         "best_rank": int(rank.min()) if rank is not None else None})
    return pd.DataFrame(rows)
