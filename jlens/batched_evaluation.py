"""Batched task and held-out evaluation with bounded vocabulary workspace.

This opt-in API does not change the legacy evaluator. Glossaries are display
metadata and must never be passed as spelling lookups. See
``docs/batched_evaluation.md`` for the notebook contract and memory limits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from jlens.evaluation import readout_position
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.metrics import (
    _LAYER_COLUMNS,
    _PAIR_COLUMNS,
    DistributionMetrics,
    _distribution_sums,
    _selected_layers,
    _special_ids,
)
from jlens.protocol import LensModel
from jlens.readout import readout, selected_token_ranks
from jlens.strict_scoring import ExplicitPromptModel

__all__ = ["evaluate_paired_batched", "evaluate_distributions_batched"]

_NAMES = ("logit lens", "J-lens")
_WORD_COLUMNS = [
    "dataset",
    "item",
    "kind",
    "word",
    "role",
    "single_token",
    "in_prompt",
    "ranks",
    "best_rank",
    "best_layer",
    "lens",
    "accepted_ids",
]
_ITEM_COLUMNS = [
    "dataset",
    "item",
    "prompt",
    "target",
    "readout_token",
    "model_top1",
    "model_correct",
    "readout_top1",
    "agreement",
    "copy_rate",
    "kl_to_final",
    "lens",
    "model_top1_id",
    "model_prediction_ranks",
    "readout_top1_ids",
    "target_ids",
    "readout_target_match",
    "n_tokens",
    "n_prompt_tokens",
    "readout_position",
]


@dataclass
class _TextSample:
    index: int
    ids: list[int]
    valid: list[int]


@dataclass
class _Sample:
    index: int
    dataset: str
    item: dict
    ids: list[int]
    position: int
    valid: list[int]
    words: list[dict]
    target_ids: tuple[int, ...]


def _prepare(model, evals, lookup, max_seq_len) -> list[_Sample]:
    tok = model.tokenizer
    special = set(getattr(tok, "all_special_ids", []) or [])
    special.update(
        token
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (token := getattr(tok, attr, None)) is not None
    )
    samples = []
    accepted = {}
    for dataset, items in evals.items():
        for i, item in enumerate(items):
            ids = model.encode_ids(item["prompt"], max_length=max_seq_len)
            valid = [p for p, token in enumerate(ids) if token not in special]
            if not valid:
                raise ValueError(f"{dataset}/{item['name']}: no non-special tokens")
            position = readout_position(tok, ids, dataset)
            prompt_ids = {ids[p] for p in valid}
            other = items[(i + len(items) // 2) % len(items)]["intermediates"]
            entries = [
                ("intermediate", word, role)
                for role, word in enumerate(item["intermediates"])
            ]
            entries += [
                ("control", word, role)
                for role, word in enumerate(other)
                if word not in item["intermediates"]
            ]
            if "target" in item:
                entries.append(("target", item["target"], 0))
            words, target_ids = [], ()
            for kind, word, role in entries:
                key = (word, dataset == "order-ops")
                if key not in accepted:
                    token_ids = tuple(sorted(lookup(*key)))
                    if any(
                        not isinstance(t, int)
                        or isinstance(t, bool)
                        or t < 0
                        or t in special
                        for t in token_ids
                    ):
                        raise ValueError("Accepted IDs must be non-special token IDs")
                    accepted[key] = token_ids
                word_ids = accepted[key]
                if kind == "target":
                    target_ids = word_ids
                words.append(
                    {
                        "dataset": dataset,
                        "item": item["name"],
                        "kind": kind,
                        "word": word,
                        "role": role,
                        "single_token": bool(word_ids),
                        "in_prompt": bool(prompt_ids.intersection(word_ids)),
                        "accepted_ids": word_ids,
                    }
                )
            samples.append(
                _Sample(
                    len(samples),
                    dataset,
                    item,
                    ids,
                    position,
                    valid,
                    words,
                    target_ids,
                )
            )
    return samples


def _batches(
    samples: list[_Sample] | list[_TextSample],
    batch_size: int,
    max_batch_tokens: int | None,
    padded: bool,
) -> Iterator[list[_Sample] | list[_TextSample]]:
    batch = []
    for sample in sorted(samples, key=lambda s: (len(s.ids), s.index)):
        length = len(sample.ids)
        if max_batch_tokens is not None and length > max_batch_tokens:
            raise ValueError(
                f"sample {sample.index}: {length} tokens exceeds "
                f"max_batch_tokens={max_batch_tokens}; increase the budget"
            )
        if batch and (
            len(batch) == batch_size
            or (not padded and length != len(batch[0].ids))
            or (
                max_batch_tokens is not None
                and length * (len(batch) + 1) > max_batch_tokens
            )
        ):
            yield batch
            batch = []
        batch.append(sample)
    if batch:
        yield batch


def _device(device: torch.device) -> None:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Batched evaluation supports CPU/CUDA, not {device}")


def _decode(model, lens, activations, layer, rows, positions, transport, matrices):
    h = activations[layer]
    _device(h.device)
    with torch.autocast(device_type=h.device.type, enabled=False):
        # Gather before casting/transport: never retain an fp32 [batch, seq, d]
        # residual or a transported activation for every layer.
        residual = h[rows.to(h.device), positions.to(h.device)].float()
        if transport:
            key = (layer, h.device)
            if key not in matrices:
                matrices[key] = lens.jacobians[layer].to(h.device, dtype=torch.float32)
            residual = residual @ matrices[key].T
    # unembed owns final norm, head, dtype/device moves and optional softcap.
    # Disable both common ambient autocast contexts for sharded HF models.
    with torch.autocast("cpu", enabled=False), torch.autocast("cuda", enabled=False):
        result = readout(model, residual)
    logits = result.logits
    _device(logits.device)
    if logits.ndim != 2 or logits.shape[0] != len(rows):
        raise ValueError("unembed must return [positions, vocabulary] logits")
    if result.ranking_scores.shape != logits.shape or result.ranking_scores.device != logits.device:
        raise ValueError("Ranking scores must match logits shape/device")
    return result


def _rank_plan(chunk, batch, word_offsets, device):
    """Small ragged accepted-ID gather, constructed once per position chunk."""
    word_indices, logit_rows, spelling_words, spelling_rows, spelling_ids = (
        [],
        [],
        [],
        [],
        [],
    )
    readout_rows, readout_owners = [], []
    for row, (owner, position, _) in enumerate(chunk):
        if position != batch[owner].position:
            continue
        readout_rows.append(row)
        readout_owners.append(owner)
        for local, word in enumerate(batch[owner].words):
            if not word["accepted_ids"]:
                continue
            w = len(word_indices)
            word_indices.append(word_offsets[owner] + local)
            logit_rows.append(row)
            for token in word["accepted_ids"]:
                spelling_words.append(w)
                spelling_rows.append(row)
                spelling_ids.append(token)
    return tuple(
        torch.tensor(x, device=device, dtype=torch.long)
        for x in (
            word_indices,
            logit_rows,
            spelling_words,
            spelling_rows,
            spelling_ids,
            readout_rows,
            readout_owners,
        )
    )


def _batch_mode(model, batching: str) -> bool:
    if any(block.training for block in model.layers):
        raise ValueError("Batched evaluation requires eval-mode blocks")
    if batching not in {"auto", "padded", "equal_length"}:
        raise ValueError("batching must be auto, padded, or equal_length")
    supports_mask = getattr(model, "supports_attention_mask", False) is True
    if batching == "padded" and not supports_mask:
        raise ValueError(
            "Adapter lacks padding-mask support; use equal_length batching"
        )
    return supports_mask and batching != "equal_length"


def _forward_batch(model, batch, padded):
    """Capture all residual blocks in one shared, optionally masked forward."""
    size, length, layers = len(batch), max(len(s.ids) for s in batch), model.n_layers
    pad = getattr(model.tokenizer, "pad_token_id", None)
    # A valid existing input ID works even when the tokenizer has no pad token.
    # Masks and true lengths, never token identity, exclude synthetic padding.
    pad = batch[0].ids[0] if pad is None else pad
    ids = torch.full((size, length), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, sample in enumerate(batch):
        ids[row, : len(sample.ids)] = torch.tensor(sample.ids)
        mask[row, : len(sample.ids)] = 1
    device = model.input_device
    _device(torch.device(device))
    with ActivationRecorder(model.layers, at=range(layers)) as recorder:
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            if padded:
                model.forward(ids.to(device), attention_mask=mask.to(device))
            else:
                model.forward(ids.to(device))
    for h in recorder.activations.values():
        if h.shape != (size, length, model.d_model):
            raise ValueError("Blocks must return [batch, sequence, d_model] residuals")
    return recorder.activations


def _run_batch(
    model, lens, batch, padded, position_chunk_size, rank_chunk_size, matrices
):
    size, layers = len(batch), model.n_layers
    activations = _forward_batch(model, batch, padded)
    word_offsets, n_words = [], 0
    coordinates = []
    for row, sample in enumerate(batch):
        word_offsets.append(n_words)
        n_words += len(sample.words)
        valid = set(sample.valid)
        coordinates.extend(
            (row, p, p in valid) for p in sorted(valid | {sample.position})
        )
    ranks = tops = model_ranks = totals = finite = None
    for start in range(0, len(coordinates), position_chunk_size):
        chunk = coordinates[start : start + position_chunk_size]
        # Index tensors stay device-side throughout this chunk's layer loop.
        indices = {}
        for h in activations.values():
            if h.device not in indices:
                indices[h.device] = (
                    torch.tensor([c[0] for c in chunk], device=h.device),
                    torch.tensor([c[1] for c in chunk], device=h.device),
                )

        def decode(layer, transport=False, indices=indices):
            rows, positions = indices[activations[layer].device]
            return _decode(
                model,
                lens,
                activations,
                layer,
                rows,
                positions,
                transport,
                matrices,
            )

        final_readout = decode(layers - 1)
        final_logits = final_readout.logits
        output_device = final_logits.device
        if ranks is None:
            ranks = torch.full(
                (2, layers, n_words), -1, dtype=torch.long, device=output_device
            )
            tops = torch.empty(
                (2, layers, size), dtype=torch.long, device=output_device
            )
            model_ranks = torch.empty_like(tops)
            totals = torch.zeros(
                (2, layers, size, 3), dtype=torch.float64, device=output_device
            )
            finite = torch.ones((), dtype=torch.bool, device=output_device)
            if any(
                t >= final_logits.shape[1]
                for s in batch
                for w in s.words
                for t in w["accepted_ids"]
            ):
                raise ValueError("Accepted token ID is outside the output vocabulary")
        elif output_device != ranks.device:
            raise ValueError("unembed must use a consistent output device")
        finite &= torch.isfinite(final_logits).all()
        final_top = final_logits.argmax(-1)
        final_logp = final_logits.float().log_softmax(-1)
        final_prob = final_logp.exp()
        owner = torch.tensor([c[0] for c in chunk], device=output_device)
        valid = torch.tensor([c[2] for c in chunk], device=output_device)
        input_tokens = torch.tensor(
            [batch[r].ids[p] for r, p, _ in chunk],
            device=output_device,
        )
        plan = _rank_plan(chunk, batch, word_offsets, output_device)
        wi, wr, sw, sr, si, rr, ro = plan
        for layer in range(layers):
            for name in range(2):
                if layer == layers - 1 and name == 1:
                    ranks[1, layer] = ranks[0, layer]
                    tops[1, layer] = tops[0, layer]
                    model_ranks[1, layer] = model_ranks[0, layer]
                    totals[1, layer] = totals[0, layer]
                    continue
                result = (
                    final_readout if layer == layers - 1 else decode(layer, name == 1)
                )
                logits, scores = result.logits, result.ranking_scores
                if logits.shape != final_logits.shape or logits.device != output_device:
                    raise ValueError("All readouts must share vocabulary shape/device")
                finite &= torch.isfinite(logits).all() & torch.isfinite(scores).all()
                top = final_top if layer == layers - 1 else logits.argmax(-1)
                logp = (
                    final_logp
                    if layer == layers - 1
                    else logits.float().log_softmax(-1)
                )
                kl = (final_prob * (final_logp - logp)).sum(-1)
                measures = torch.stack(
                    (
                        (top == final_top).double(),
                        (top == input_tokens).double(),
                        kl.double(),
                    ),
                    dim=-1,
                )
                totals[name, layer].index_add_(0, owner, measures * valid[:, None])
                tops[name, layer, ro] = top[rr]
                for offset in range(0, len(rr), rank_chunk_size):
                    readout_rows = rr[offset : offset + rank_chunk_size]
                    readout_owners = ro[offset : offset + rank_chunk_size]
                    model_ranks[name, layer, readout_owners] = selected_token_ranks(
                        scores, final_top[readout_rows, None], row_indices=readout_rows,
                    ).squeeze(-1)
                if len(wi):
                    best = torch.full(
                        (len(wi),), -torch.inf, device=output_device, dtype=scores.dtype
                    )
                    spelling_scores = scores[sr, si]
                    best.scatter_reduce_(0, sw, spelling_scores, reduce="amax")
                    # Pick the earliest ID among equally best accepted spellings.
                    best_ids = torch.full_like(wi, scores.shape[-1])
                    candidates = torch.where(spelling_scores == best[sw], si, scores.shape[-1])
                    best_ids.scatter_reduce_(0, sw, candidates, reduce="amin")
                    for offset in range(0, len(wi), rank_chunk_size):
                        sl = slice(offset, offset + rank_chunk_size)
                        ranked = selected_token_ranks(
                            scores, best_ids[sl, None], row_indices=wr[sl],
                        )
                        ranks[name, layer, wi[sl]] = ranked.squeeze(-1)
                del result, logits, scores, logp, top, kl, measures
        del final_readout, final_logits, final_logp, final_prob, final_top
    finite &= torch.isfinite(totals).all()
    # Only batch-level synchronizations; no per-item or per-word GPU readback.
    if not bool(finite.cpu()):
        raise ValueError("Nonfinite logits or metrics in batched evaluation")
    counts = torch.tensor([len(s.valid) for s in batch], device=totals.device)
    totals /= counts[None, None, :, None]
    return (
        ranks.cpu().numpy(),
        tops.cpu().numpy(),
        totals.cpu().numpy(),
        word_offsets,
        model_ranks.cpu().numpy(),
    )


@torch.inference_mode()
def evaluate_paired_batched(
    model: LensModel,
    lens: JacobianLens,
    evals: dict[str, list[dict]],
    desc: str = "batched paired lenses",
    *,
    spelling_lookup: Callable[[str, bool], set[int]],
    bos_policy: str | None = None,
    preserve_prompt_whitespace: bool = True,
    batch_size: int = 8,
    max_batch_tokens: int | None = 2048,
    max_seq_len: int = 512,
    position_chunk_size: int = 8,
    rank_chunk_size: int = 8,
    batching: str = "auto",
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict ``(words, items)`` tables from one shared model pass per batch.

    Supply ``DecodedSpellings`` as ``spelling_lookup`` and either an
    ``ExplicitPromptModel`` or an explicit ``bos_policy``. Empty accepted sets
    produce null ranks/correctness, never prefix fallbacks. Target ranks and
    argmax matches use exactly the same IDs (including order-ops synonyms).
    Whitespace is preserved, no chat template is applied, and overlength
    prompts fail rather than truncate. Legacy stripping is deliberately not
    available in this new API.

    ``auto`` right-pads length-sorted batches only if the adapter advertises
    ``supports_attention_mask=True``; otherwise it groups exact equal lengths.
    ``padded`` requires that capability; ``equal_length`` never pads or passes
    masks. HFLensModel advertises it only for an explicit decoder mask API.
    Custom adapters must correctly support independent batched causal rows;
    advertising mask support additionally promises right-padding semantics.
    Models must be in eval mode. CPU/CUDA only; no persistent KV caches.

    ``max_batch_tokens`` caps batch_size * longest encoded length, including
    BOS/padding. Vocabulary workspace is O(max(position_chunk_size,
    rank_chunk_size) * vocab), not O(layers * batch * sequence * vocab).
    Full batch block activations and one fp32 J per inner layer/device are
    retained; the latter are cached for this call, not repeatedly transferred.
    Reductions are fp32 distributions/fp64 sums, with autocast disabled;
    caller TF32 settings and model parameter dtypes are unchanged. Transport
    activations are O(position_chunk_size * hidden_width), not cached per layer.
    Nonfinite readouts fail explicitly.

    Existing columns match strict ``evaluate_paired``; additional numeric-ID
    columns make argmax target-hit curves independent of display/gloss text.
    ``model_prediction_ranks`` tracks the exact final model argmax ID's rank
    in pre-softcap lexical scores at each block, independently of targets.
    Ranks break ties by ascending token ID; the final prediction's lexical rank
    need not be 1 if softcapping tied distinct head scores. Actual argmax,
    correctness, agreement, copy rate and KL still use distribution logits.
    Final logits, ranks, correctness and behavioral metrics are shared exactly
    between lenses, ignoring any fitted final-layer J. Correctness is therefore
    independent of the lens, but depends on the adapter's final readout.
    See docs/batched_evaluation.md for schema, aggregation and limitations.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("max_seq_len", max_seq_len),
        ("position_chunk_size", position_chunk_size),
        ("rank_chunk_size", rank_chunk_size),
        ("max_batch_tokens", max_batch_tokens),
    ):
        if value is None and name == "max_batch_tokens":
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not preserve_prompt_whitespace:
        raise ValueError("Strict batched evaluation always preserves prompt whitespace")
    if not callable(spelling_lookup):
        raise TypeError("spelling_lookup must be a strict whole-token callable")
    if bos_policy is not None:
        if isinstance(model, ExplicitPromptModel) and model.bos_policy != bos_policy:
            raise ValueError("bos_policy conflicts with ExplicitPromptModel")
        model = ExplicitPromptModel(model, bos_policy=bos_policy)
    elif not isinstance(model, ExplicitPromptModel):
        raise ValueError("Supply ExplicitPromptModel or an explicit bos_policy")
    if model.n_layers < 1 or lens.d_model != model.d_model:
        raise ValueError("Invalid model layer count or mismatched lens d_model")
    missing = set(range(model.n_layers - 1)) - set(lens.source_layers)
    if missing:
        raise ValueError(f"J-lens is missing inner layers {sorted(missing)}")
    for layer in range(model.n_layers - 1):
        if lens.jacobians[layer].shape != (model.d_model, model.d_model):
            raise ValueError(f"Invalid Jacobian shape at layer {layer}")
    padded = _batch_mode(model, batching)
    samples = _prepare(model, evals, spelling_lookup, max_seq_len)
    batches = list(_batches(samples, batch_size, max_batch_tokens, padded))
    results, matrices = {}, {}
    tok = model.tokenizer

    def decode(token):
        return tok.decode([int(token)], clean_up_tokenization_spaces=False)

    for batch in tqdm(batches, desc=desc, disable=not progress, unit="batch"):
        ranks, tops, totals, offsets, model_ranks = _run_batch(
            model,
            lens,
            batch,
            padded,
            position_chunk_size,
            rank_chunk_size,
            matrices,
        )
        for i, sample in enumerate(batch):
            word_rows, item_rows = [], []
            model_top = int(tops[0, -1, i])
            correct = model_top in sample.target_ids if sample.target_ids else None
            for n, name in enumerate(_NAMES):
                for j, word in enumerate(sample.words):
                    rank = (
                        ranks[n, :, offsets[i] + j].copy()
                        if word["single_token"]
                        else None
                    )
                    word_rows.append(
                        {
                            **word,
                            "lens": name,
                            "ranks": rank,
                            "best_rank": int(rank.min()) if rank is not None else None,
                            "best_layer": int(rank.argmin())
                            if rank is not None
                            else None,
                        }
                    )
                top_ids = tops[n, :, i].copy()
                item_rows.append(
                    {
                        "dataset": sample.dataset,
                        "item": sample.item["name"],
                        "prompt": sample.item["prompt"],
                        "target": sample.item.get("target"),
                        "readout_token": decode(sample.ids[sample.position]),
                        "model_top1": decode(model_top),
                        "model_correct": correct,
                        "readout_top1": [decode(t) for t in top_ids],
                        "lens": name,
                        "agreement": totals[n, :, i, 0].copy(),
                        "copy_rate": totals[n, :, i, 1].copy(),
                        "kl_to_final": totals[n, :, i, 2].copy(),
                        "model_top1_id": model_top,
                        "model_prediction_ranks": model_ranks[n, :, i].copy(),
                        "readout_top1_ids": top_ids,
                        "target_ids": sample.target_ids,
                        "readout_target_match": (
                            np.isin(top_ids, sample.target_ids)
                            if sample.target_ids
                            else None
                        ),
                        "n_tokens": len(sample.valid),
                        "n_prompt_tokens": len(sample.ids),
                        "readout_position": sample.position,
                    }
                )
            results[sample.index] = word_rows, item_rows
    # Restore original dataset/item/lens order after length sorting.
    words, items = [], []
    for index in range(len(samples)):
        w, i = results[index]
        words.extend(w)
        items.extend(i)
    frames = (
        pd.DataFrame(words, columns=_WORD_COLUMNS),
        pd.DataFrame(items, columns=_ITEM_COLUMNS),
    )
    batching_info = {
        "batching": "padded" if padded else "equal_length",
        "n_model_passes": len(batches),
        "n_items": len(samples),
        "batch_sizes": [len(batch) for batch in batches],
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "max_seq_len": max_seq_len,
        "position_chunk_size": position_chunk_size,
        "rank_chunk_size": rank_chunk_size,
        "bos_policy": model.bos_policy,
    }
    for frame in frames:
        frame.attrs["evaluation"] = batching_info.copy()
    return frames


def _run_distribution_batch(model, lens, batch, padded, chunk_size, matrices):
    activations = _forward_batch(model, batch, padded)
    coordinates = [(row, p) for row, sample in enumerate(batch) for p in sample.valid]
    layers = model.n_layers
    totals = pairs = finite = reference = None
    for start in range(0, len(coordinates), chunk_size):
        chunk = coordinates[start : start + chunk_size]
        indices = {
            device: (
                torch.tensor([c[0] for c in chunk], device=device),
                torch.tensor([c[1] for c in chunk], device=device),
            )
            for device in {h.device for h in activations.values()}
        }

        def decode(layer, transported=False, indices=indices):
            nonlocal finite, reference
            rows, positions = indices[activations[layer].device]
            logits = _decode(
                model, lens, activations, layer, rows, positions, transported, matrices
            ).logits
            shape_device = (logits.shape[1], logits.device)
            if reference is None:
                reference = shape_device
                finite = torch.ones((), dtype=torch.bool, device=logits.device)
            elif shape_device != reference:
                raise ValueError("All readouts must share vocabulary shape/device")
            finite &= torch.isfinite(logits).all()
            top = logits.argmax(-1)
            logp = logits.float().log_softmax(-1)
            finite &= torch.isfinite(logp).all()
            return logp, top

        final_logp, final_top = decode(layers - 1)
        final_prob = final_logp.exp()
        if totals is None:
            totals = torch.zeros(
                (2, layers, 3), dtype=torch.float64, device=final_logp.device
            )
            pairs = torch.zeros(
                (layers, 2), dtype=torch.float64, device=final_logp.device
            )
        for layer in range(layers):
            logps, tops = [], []
            for transported in (False, True):
                logp, top = (
                    (final_logp, final_top)
                    if layer == layers - 1
                    else decode(layer, transported)
                )
                logps.append(logp)
                tops.append(top)
            probs = (
                [final_prob, final_prob]
                if layer == layers - 1
                else [p.exp() for p in logps]
            )
            for name in range(2):
                totals[name, layer] += _distribution_sums(
                    logps[name],
                    probs[name],
                    tops[name],
                    final_logp,
                    final_prob,
                    final_top,
                )
            pairs[layer, 0] += 0.5 * (
                (probs[0] - probs[1]) * (logps[0] - logps[1])
            ).sum(dtype=torch.float64)
            pairs[layer, 1] += (tops[0] == tops[1]).sum(dtype=torch.float64)
            del logp, top, logps, tops, probs
        del final_logp, final_prob, final_top
    finite &= torch.isfinite(totals).all() & torch.isfinite(pairs).all()
    if not bool(finite.cpu()):
        raise ValueError(
            "Nonfinite logits or metrics in batched distribution evaluation"
        )
    return totals.cpu(), pairs.cpu()


@torch.inference_mode()
def evaluate_distributions_batched(
    model: LensModel,
    lens: JacobianLens,
    texts: Iterable[str],
    *,
    batch_size: int = 8,
    max_batch_tokens: int | None = 2048,
    max_seq_len: int = 512,
    position_chunk_size: int = 8,
    batching: str = "auto",
    progress: bool = True,
) -> DistributionMetrics:
    """Held-out token-weighted distributions, one shared forward per batch.

    Returns the existing ``DistributionMetrics.layers`` / ``.pairs`` schema
    over all blocks, with only SAME-layer logit-lens vs J-lens pairs. Each
    non-special, non-padding input position contributes equally, including the
    last real position. Empty/special-only texts count but are not forwarded;
    a corpus without valid positions fails. KL/entropy are in nats, and
    symmetric KL is the half-sum. Final readouts are shared exactly.

    Encoding preserves the adapter's policy: ``ExplicitPromptModel`` preserves
    whitespace/BOS and rejects overlength texts; bare HF adapters truncate as
    in ``encode``. CPU ``encode_ids`` is preferred; other adapters fall back to
    ``encode``. No packing, chat template, target scoring or glossary is used.

    Batch size, padded-token budget and ``auto/padded/equal_length`` semantics
    match ``evaluate_paired_batched``. CPU/CUDA eval-mode models are required.
    HF decoders receive right-padding masks; unknown mask APIs use equal-length
    groups instead. Custom adapters must honor independent batched causal rows.

    Only final and current-layer distributions coexist: O(chunk_size * vocab)
    readout workspace. All batch block activations and cached fp32 Jacobians
    remain resident. Distributions are fp32; aggregate sums fp64. Autocast is
    disabled; model parameter dtypes and caller TF32 settings are unchanged.
    Transport activations are O(chunk_size * hidden_width), not cached per layer.
    Nonfinite readouts fail. Inputs
    are tokenized/materialized for length sorting; no corpus logits are kept.
    Both tables include batch provenance in ``attrs['evaluation']``.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("max_batch_tokens", max_batch_tokens),
        ("max_seq_len", max_seq_len),
        ("position_chunk_size", position_chunk_size),
    ):
        if name == "max_batch_tokens" and value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of strings, not one string")
    selected = _selected_layers(model, lens, None)
    padded = _batch_mode(model, batching)
    specials = _special_ids(model.tokenizer)
    encode_ids = getattr(model, "encode_ids", None)
    samples, n_texts = [], 0
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"text {index} is not a string")
        n_texts += 1
        if callable(encode_ids):
            ids = list(encode_ids(text, max_length=max_seq_len))
        else:
            encoded = model.encode(text, max_length=max_seq_len)
            _device(encoded.device)
            if encoded.ndim != 2 or encoded.shape[0] != 1:
                raise ValueError("encode must return [1, seq_len] input_ids")
            ids = encoded[0].tolist()
        valid = [p for p, token in enumerate(ids) if token not in specials]
        if valid:
            samples.append(_TextSample(index, ids, valid))
    n_tokens = sum(len(s.valid) for s in samples)
    if not n_tokens:
        raise ValueError("held-out corpus has no non-special positions")
    batches = list(_batches(samples, batch_size, max_batch_tokens, padded))
    totals = torch.zeros((2, len(selected), 3), dtype=torch.float64)
    pairs = torch.zeros((len(selected), 2), dtype=torch.float64)
    matrices = {}
    for batch in tqdm(
        batches,
        desc="batched held-out lens metrics",
        disable=not progress,
        unit="batch",
    ):
        batch_totals, batch_pairs = _run_distribution_batch(
            model, lens, batch, padded, position_chunk_size, matrices
        )
        totals += batch_totals
        pairs += batch_pairs
    totals /= n_tokens
    pairs /= n_tokens
    counts = dict(
        n_tokens=n_tokens, n_texts=n_texts, n_texts_used=len(samples), weighting="token"
    )
    layer_rows = [
        dict(
            lens=name,
            layer=layer,
            is_final=layer == model.n_layers - 1,
            kl_model_to_lens=float(totals[n, layer, 0]),
            entropy=float(totals[n, layer, 1]),
            top1_agreement=float(totals[n, layer, 2]),
            **counts,
        )
        for n, name in enumerate(_NAMES)
        for layer in selected
    ]
    pair_rows = [
        dict(
            lens_a=_NAMES[0],
            layer_a=layer,
            lens_b=_NAMES[1],
            layer_b=layer,
            symmetric_kl=float(pairs[layer, 0]),
            top1_agreement=float(pairs[layer, 1]),
            **counts,
        )
        for layer in selected
    ]
    result = DistributionMetrics(
        pd.DataFrame(layer_rows, columns=_LAYER_COLUMNS),
        pd.DataFrame(pair_rows, columns=_PAIR_COLUMNS),
    )
    info = dict(
        batching="padded" if padded else "equal_length",
        n_model_passes=len(batches),
        batch_sizes=[len(b) for b in batches],
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_seq_len=max_seq_len,
        position_chunk_size=position_chunk_size,
        bos_policy=getattr(model, "bos_policy", "adapter"),
        **counts,
    )
    for frame in (result.layers, result.pairs):
        frame.attrs["evaluation"] = info.copy()
    return result
