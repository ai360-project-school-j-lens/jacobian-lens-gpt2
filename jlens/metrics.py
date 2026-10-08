# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Held-out distribution metrics and linear-head geometry (no plotting).

These are adaptations suitable for Figures 55/56-style analyses, not claims
of reproducing a reference experiment. See ``docs/lens_metrics.md`` for exact
schemas, weighting, decoder conventions, and memory/precision limitations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "DistributionMetrics",
    "evaluate_distributions",
    "lens_vector_geometry",
    "spearman_lens_vs_logits",
]

_NAMES = ("logit lens", "J-lens")
_LAYER_COLUMNS = [
    "lens",
    "layer",
    "is_final",
    "kl_model_to_lens",
    "entropy",
    "top1_agreement",
    "n_tokens",
    "n_texts",
    "n_texts_used",
    "weighting",
]
_PAIR_COLUMNS = [
    "lens_a",
    "layer_a",
    "lens_b",
    "layer_b",
    "symmetric_kl",
    "top1_agreement",
    "n_tokens",
    "n_texts",
    "n_texts_used",
    "weighting",
]
_GEOMETRY_COLUMNS = [
    "layer",
    "is_final",
    "status",
    "reason",
    "mean_cosine",
    "n_vocab",
    "n_valid_vectors",
    "n_zero_vectors",
    "weighting",
    "convention",
]


@dataclass(frozen=True)
class DistributionMetrics:
    """Aggregate ``layers`` and same-layer, cross-lens ``pairs`` DataFrames.

    KL and entropy use natural logarithms (nats); agreements are fractions.
    Counts and the string ``weighting='token'`` appear in every row. No logits
    or text contents are retained. See :func:`evaluate_distributions`.
    """

    layers: pd.DataFrame
    pairs: pd.DataFrame


def _selected_layers(
    model: LensModel, lens: JacobianLens, layers: Sequence[int] | None
) -> list[int]:
    if model.n_layers < 1 or lens.d_model != model.d_model:
        raise ValueError("model must have layers and lens/model d_model must match")
    final = model.n_layers - 1
    requested = list(range(model.n_layers) if layers is None else layers)
    if any(
        isinstance(i, bool) or not isinstance(i, int) or not 0 <= i <= final
        for i in requested
    ):
        raise ValueError(
            "layers must be integer block indices in model range (not bool)"
        )
    selected = sorted(set(requested) | {final})
    missing = set(selected) - {final} - set(lens.jacobians)
    if missing:
        raise ValueError(f"J-lens is missing inner layers {sorted(missing)}")
    for i in selected:
        if i != final:
            matrix = lens.jacobians[i]
            if matrix.shape != (model.d_model, model.d_model):
                raise ValueError(f"layer {i}: invalid Jacobian shape")
            _finite(matrix, f"layer {i} Jacobian")
    return selected


def _validate_device(device: torch.device) -> None:
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"lens metrics support only CPU/CUDA compute devices; got {device}. "
            "Float64 reductions are required."
        )


def _finite(value: torch.Tensor, context: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{context}: nonfinite values")


def _special_ids(tokenizer) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    ids.update(
        value
        for name in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (value := getattr(tokenizer, name, None)) is not None
    )
    return ids


def _decode(
    model: LensModel,
    lens: JacobianLens,
    activation: torch.Tensor,
    positions: list[int],
    layer: int,
    transported: bool,
    text_index: int,
    matrices: dict[tuple[int, torch.device], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_device(activation.device)
    residual = activation[0, positions].float()
    with torch.autocast(device_type=residual.device.type, enabled=False):
        if transported:
            key = (layer, residual.device)
            if key not in matrices:
                matrices[key] = lens.jacobians[layer].to(
                    device=residual.device, dtype=torch.float32
                )
            residual = residual @ matrices[key].T
        _finite(residual, f"text {text_index}, layer {layer}: residual")
        logits = model.unembed(residual)
    _validate_device(logits.device)
    if logits.ndim != 2 or logits.shape[0] != len(positions):
        raise ValueError("unembed must return [n_positions, vocab_size]")
    _finite(logits, f"text {text_index}, layer {layer}, J={transported}")
    # Normalization can round distinct logits into fp32 ties. Preserve the
    # decoder's ordering before casting/normalizing for distribution arithmetic.
    top1 = logits.argmax(-1)
    logp = logits.float().log_softmax(-1)
    _finite(logp, f"text {text_index}, layer {layer}: log probabilities")
    return logp, top1


def _distribution_sums(
    logp: torch.Tensor,
    prob: torch.Tensor,
    top1: torch.Tensor,
    final_logp: torch.Tensor,
    final_prob: torch.Tensor,
    final_top1: torch.Tensor,
) -> torch.Tensor:
    """Three scalar sums; no retained vocabulary-sized intermediates."""
    return torch.stack(
        [
            (final_prob * (final_logp - logp)).sum(dtype=torch.float64),
            -(prob * logp).sum(dtype=torch.float64),
            (top1 == final_top1).sum(dtype=torch.float64),
        ]
    )


@torch.inference_mode()
def evaluate_distributions(
    model: LensModel,
    lens: JacobianLens,
    texts: Iterable[str],
    *,
    layers: Sequence[int] | None = None,
    max_seq_len: int = 512,
    position_chunk_size: int = 8,
    pairwise: bool = True,
    progress: bool = True,
    desc: str = "held-out lens metrics",
) -> DistributionMetrics:
    """Compare both lenses on one model forward per supplied held-out text.

    ``layers`` defaults to every block; the final block is always included.
    Inner requested layers must be fitted. Both lenses use the same raw block
    outputs and adapter ``unembed`` (including final normalization/softcap).
    J-lens applies ``h.float() @ J.T`` first. Final outputs are shared exactly,
    ignoring any fitted final-layer matrix, as in ``evaluate_paired``.

    Each non-special *input* position contributes one observation, including
    the last position (no next-token label is required). Texts with no valid
    positions are counted but skipped. An empty valid corpus raises ValueError.
    Truncation is per text through ``model.encode``; there is no packing or
    sliding window. Caller supplies independent held-out texts and an eval-mode
    model satisfying LensModel's deterministic-forward contract.

    ``layers`` table: one row per (lens, layer), KL(model || lens), lens entropy,
    and top-1 agreement with the model. ``pairs``: one logit-lens vs J-lens
    comparison per selected layer, in ascending layer order (no cross-layer
    or self pairs); symmetric_kl = 0.5 * (KL(a || b) + KL(b || a)). Disable with
    ``pairwise=False`` to return an empty pairs table with the same schema.
    All means are token
    weighted, not means of text means. Top-1 uses raw decoder logits, with
    torch.argmax's first index for ties, not rounded log-probabilities.

    Layers are streamed: only the final and current layer's two vocabulary
    distributions are held, O(position_chunk_size * vocab_size) memory plus
    temporaries, independent of layer count. Chunking is over positions, NOT
    vocabulary; every decode includes the entire vocabulary. One text's block
    activations are retained. Requested Jacobians are cached on activation
    devices for this call (O(n_layers * d_model**2) additional storage).
    Pairwise work and aggregate storage are linear in selected layer count.
    Only CPU/CUDA compute devices are supported (validated explicitly);
    reductions use fp32 distributions and fp64 sums. Adapter handles model
    dtype/device; autocast is disabled for readout and transport, but caller
    TF32 settings are unchanged. Nonfinite readouts/metrics raise ValueError.
    """
    if max_seq_len < 1 or position_chunk_size < 1:
        raise ValueError("max_seq_len and position_chunk_size must be positive")
    if isinstance(texts, str):
        raise TypeError("texts must be an iterable of strings, not one string")
    selected = _selected_layers(model, lens, layers)
    final = model.n_layers - 1
    keys = [(name, layer) for name in _NAMES for layer in selected]
    n_readouts = len(keys)
    # Small aggregate buffers only; float64 prevents long-corpus summation drift.
    totals = torch.zeros(n_readouts, 3, dtype=torch.float64)
    pair_totals = (
        torch.zeros(len(selected), 2, dtype=torch.float64) if pairwise else None
    )
    n_tokens = n_texts = n_texts_used = 0
    specials = _special_ids(model.tokenizer)
    matrices: dict[tuple[int, torch.device], torch.Tensor] = {}

    for text_index, text in enumerate(tqdm(texts, desc=desc, disable=not progress)):
        if not isinstance(text, str):
            raise TypeError(f"text {text_index} is not a string")
        n_texts += 1
        input_ids = model.encode(text, max_length=max_seq_len)
        _validate_device(input_ids.device)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("encode must return [1, seq_len] input_ids")
        positions = [
            i for i, token in enumerate(input_ids[0].tolist()) if token not in specials
        ]
        if not positions:
            continue
        n_texts_used += 1
        n_tokens += len(positions)
        with ActivationRecorder(model.layers, at=selected) as recorder:
            # Disable ambient autocast, without changing the model's stored dtype.
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                model.forward(input_ids)
        for start in range(0, len(positions), position_chunk_size):
            chunk = positions[start : start + position_chunk_size]

            final_logp, final_top1 = _decode(
                model,
                lens,
                recorder.activations[final],
                chunk,
                final,
                False,
                text_index,
                matrices,
            )
            final_prob = final_logp.exp()
            chunk_totals = torch.empty_like(totals, device=final_logp.device)
            chunk_pairs = (
                torch.empty_like(pair_totals, device=final_logp.device)
                if pairwise
                else None
            )
            for i, layer in enumerate(selected):
                # Keep only this layer's two readouts, plus the shared reference.
                logps, top1s = [], []
                for transported in (False, True):
                    logp, top1 = (
                        (final_logp, final_top1)
                        if layer == final
                        else _decode(
                            model,
                            lens,
                            recorder.activations[layer],
                            chunk,
                            layer,
                            transported,
                            text_index,
                            matrices,
                        )
                    )
                    if (
                        logp.shape != final_logp.shape
                        or logp.device != final_logp.device
                    ):
                        raise ValueError(
                            "all readouts must share vocabulary shape and device"
                        )
                    logps.append(logp)
                    top1s.append(top1)
                probs = [p.exp() for p in logps] if layer != final else [final_prob] * 2
                for j in range(2):
                    chunk_totals[j * len(selected) + i] = _distribution_sums(
                        logps[j],
                        probs[j],
                        top1s[j],
                        final_logp,
                        final_prob,
                        final_top1,
                    )
                if pairwise:
                    # Half-sum with exact zero for identical readouts; no matmul.
                    chunk_pairs[i, 0] = 0.5 * (
                        (probs[0] - probs[1]) * (logps[0] - logps[1])
                    ).sum(dtype=torch.float64)
                    chunk_pairs[i, 1] = (top1s[0] == top1s[1]).sum(dtype=torch.float64)
                del logp, top1, logps, probs, top1s
            _finite(chunk_totals, f"text {text_index}: metrics")
            totals += chunk_totals.cpu()
            if pairwise:
                _finite(chunk_pairs, f"text {text_index}: pairwise metrics")
                pair_totals += chunk_pairs.cpu()
            del chunk_totals, chunk_pairs, final_logp, final_prob, final_top1
        del recorder
    if not n_tokens:
        raise ValueError("held-out corpus has no non-special positions")
    counts = dict(
        n_tokens=n_tokens, n_texts=n_texts, n_texts_used=n_texts_used, weighting="token"
    )
    layer_rows = [
        dict(
            lens=name,
            layer=layer,
            is_final=layer == final,
            kl_model_to_lens=float(totals[i, 0] / n_tokens),
            entropy=float(totals[i, 1] / n_tokens),
            top1_agreement=float(totals[i, 2] / n_tokens),
            **counts,
        )
        for i, (name, layer) in enumerate(keys)
    ]
    pair_rows = (
        [
            dict(
                lens_a=_NAMES[0],
                layer_a=layer,
                lens_b=_NAMES[1],
                layer_b=layer,
                symmetric_kl=float(pair_totals[i, 0] / n_tokens),
                top1_agreement=float(pair_totals[i, 1] / n_tokens),
                **counts,
            )
            for i, layer in enumerate(selected)
        ]
        if pairwise
        else []
    )
    return DistributionMetrics(
        pd.DataFrame(layer_rows, columns=_LAYER_COLUMNS),
        pd.DataFrame(pair_rows, columns=_PAIR_COLUMNS),
    )


@torch.inference_mode()
def lens_vector_geometry(
    model: LensModel,
    lens: JacobianLens,
    *,
    layers: Sequence[int] | None = None,
    unembedding_weight: torch.Tensor | None = None,
    vocab_chunk_size: int = 4096,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Vocabulary-mean cosine of corresponding rows of W_U and W_U @ J_l.

    This is explicitly **linear-head-only** geometry, NOT the effective linear
    map of ``model.unembed``: normalization, bias, and softcapping are excluded.
    The actual decoded lens in :func:`evaluate_distributions` includes them.
    Final layer uses identity regardless of stored Jacobians.

    Pass ``unembedding_weight`` with shape [vocab, d_model] for arbitrary protocol
    implementations. Otherwise inspect ``_lm_head`` then ``lm_head`` and accept
    only a plain nn.Linear (not a quantized/custom nonlinear head). If no head
    is accessible, return rows with status='unsupported' and a reason, rather
    than probing/linearizing ``unembed`` or mistaking embeddings for the head.

    Compute in fp32 on CPU/CUDA ``device`` (default weight device); other
    compute devices raise ValueError because reductions require fp64. Transfer only
    vocabulary chunks plus one layer matrix. Zero-length rows in either vector
    are excluded from the mean and counted; if every row is zero the status is
    'undefined' and mean_cosine is NaN. Nonfinite weights/results raise ValueError.
    All vocabulary entries, including special tokens, have equal weight.
    """
    if vocab_chunk_size < 1:
        raise ValueError("vocab_chunk_size must be positive")
    selected = _selected_layers(model, lens, layers)
    final = model.n_layers - 1
    weight = unembedding_weight
    if weight is None:
        head = getattr(model, "_lm_head", None)
        if head is None:
            head = getattr(model, "lm_head", None)
        if type(head) is nn.Linear:
            weight = head.weight
    base = dict(weighting="vocabulary", convention="linear_head_only: W_U vs W_U @ J")
    if weight is None:
        return pd.DataFrame(
            [
                dict(
                    layer=i,
                    is_final=i == final,
                    status="unsupported",
                    reason="No accessible plain linear head; pass unembedding_weight explicitly",
                    mean_cosine=float("nan"),
                    n_vocab=0,
                    n_valid_vectors=0,
                    n_zero_vectors=0,
                    **base,
                )
                for i in selected
            ],
            columns=_GEOMETRY_COLUMNS,
        )
    if weight.ndim != 2 or weight.shape[1] != model.d_model or not weight.shape[0]:
        raise ValueError("unembedding_weight must have shape [vocab_size > 0, d_model]")
    target = torch.device(device) if device is not None else weight.device
    _validate_device(target)
    rows = []
    for layer in tqdm(selected, desc="lens vector geometry", disable=not progress):
        matrix = None if layer == final else lens.jacobians[layer].to(target).float()
        cosine_sum = torch.zeros((), dtype=torch.float64, device=target)
        n_valid = torch.zeros((), dtype=torch.int64, device=target)
        with torch.autocast(device_type=target.type, enabled=False):
            for start in range(0, weight.shape[0], vocab_chunk_size):
                original = weight[start : start + vocab_chunk_size].to(target).float()
                transformed = original if matrix is None else original @ matrix
                _finite(original, f"layer {layer}: unembedding weight")
                _finite(transformed, f"layer {layer}: transformed vectors")
                norm_a = torch.linalg.vector_norm(original, dim=-1)
                norm_b = torch.linalg.vector_norm(transformed, dim=-1)
                _finite(norm_a, f"layer {layer}: vector norms")
                _finite(norm_b, f"layer {layer}: vector norms")
                valid = (norm_a > 0) & (norm_b > 0)
                cosine = (
                    (
                        (original[valid] / norm_a[valid, None])
                        * (transformed[valid] / norm_b[valid, None])
                    )
                    .sum(-1)
                    .clamp(-1, 1)
                )
                if matrix is None:
                    cosine = torch.ones_like(cosine)
                _finite(cosine, f"layer {layer}: cosines")
                cosine_sum += cosine.sum(dtype=torch.float64)
                n_valid += valid.sum()
        count = int(n_valid.item())
        rows.append(
            dict(
                layer=layer,
                is_final=layer == final,
                status="ok" if count else "undefined",
                reason=""
                if count
                else "All corresponding vector pairs contain a zero vector",
                mean_cosine=float(cosine_sum.item() / count) if count else float("nan"),
                n_vocab=weight.shape[0],
                n_valid_vectors=count,
                n_zero_vectors=weight.shape[0] - count,
                **base,
            )
        )
        del matrix
    return pd.DataFrame(rows, columns=_GEOMETRY_COLUMNS)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """1-based ranks of a 1-D tensor with ties averaged (scipy's ``average`` method)."""
    ranks = torch.empty_like(values)
    ranks[values.argsort()] = torch.arange(
        1, values.numel() + 1, dtype=values.dtype, device=values.device
    )
    unique, inverse = values.unique(return_inverse=True)
    zeros = torch.zeros(unique.numel(), dtype=values.dtype, device=values.device)
    totals = zeros.index_add(0, inverse, ranks)
    counts = zeros.index_add(0, inverse, torch.ones_like(ranks))
    return (totals / counts)[inverse]


def spearman_lens_vs_logits(
    lens_scores: torch.Tensor,
    model_logits: torch.Tensor,
    candidate_ids: Sequence[int],
) -> float:
    """Spearman correlation between a lens readout and the model's own next-token
    logits, restricted to a candidate answer set.

    The paper's verbal-report metric: at the position just before the model names its
    answer, does the lens rank the candidate answers the way the model's output
    distribution does? Restricting to a candidate set is what makes the number
    meaningful -- over the full vocabulary the correlation is dominated by the
    ordering of tokens that are irrelevant to the question.

    Args:
        lens_scores: ``[vocab]`` lens logits at the readout position.
        model_logits: ``[vocab]`` model logits at the same position.
        candidate_ids: Vocabulary ids to correlate over. Duplicates are dropped and
            the given order is irrelevant to the result.

    Returns:
        Spearman rho, or NaN when fewer than two distinct candidates are given or
        either side is constant across them.
    """
    ids = list(dict.fromkeys(int(i) for i in candidate_ids))
    if len(ids) < 2:
        return float("nan")
    index = torch.tensor(ids, device=lens_scores.device)
    lens_rank = _average_ranks(lens_scores.float().flatten()[index])
    model_rank = _average_ranks(
        model_logits.float().flatten()[index.to(model_logits.device)]
    ).to(lens_rank.device)
    left = lens_rank - lens_rank.mean()
    right = model_rank - model_rank.mean()
    denominator = left.norm() * right.norm()
    if denominator == 0:
        return float("nan")
    return float((left * right).sum() / denominator)
