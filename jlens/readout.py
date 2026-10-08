"""Separate lexical ordering from the model's finite-precision distribution.

Distribution logits include model-specific transforms (such as softcapping).
Ranking scores precede monotone saturating transforms. Both use the model's
native head precision; this does not reconstruct precision lost in the head.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product

import torch

from jlens.protocol import LensModel


@dataclass(frozen=True)
class LensReadout:
    """Matching ``[..., vocab]`` distribution logits and lexical scores.

    Probabilities, KL, entropy and actual model predictions must use ``logits``.
    Lexical ranks and lexical top-k displays use ``ranking_scores``. They can
    disagree when the model's finite-precision transform creates ties.
    """

    logits: torch.Tensor
    ranking_scores: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.logits.shape != self.ranking_scores.shape
            or self.logits.device != self.ranking_scores.device
        ):
            raise ValueError("Logits and ranking scores must match shape and device")


def as_readout(value: torch.Tensor | LensReadout) -> LensReadout:
    """Legacy tensors supply both spaces; lost ordering cannot be recovered."""
    return value if isinstance(value, LensReadout) else LensReadout(value, value)


def warn_legacy_readout(model: LensModel) -> None:
    """Flag known softcapped tensor-only readouts without changing their API."""
    if getattr(model, "_logit_softcap", None) is not None:
        warnings.warn(
            "Tensor-only readout on a softcapped model: lexical ranks use "
            "rounded distribution logits; pre-softcap ordering is unavailable. "
            "Return LensReadout (or pass return_readout=True to the built-in "
            "lens helpers) for lexical ranking. Exact ties use token-ID order.",
            UserWarning,
            stacklevel=3,
        )


def readout(model: LensModel, residual: torch.Tensor) -> LensReadout:
    """Use the optional dual-readout capability, or the legacy unembed contract.

    Custom adapters with saturating transforms should implement
    ``unembed_readout(residual) -> LensReadout``. No new required protocol member
    or keyword argument to existing ``unembed`` implementations is introduced.
    """
    method = getattr(model, "unembed_readout", None)
    return as_readout(method(residual) if method is not None else model.unembed(residual))


# Bound temporary storage independently of vocabulary, position and target counts.
_ROW_CHUNK = 32
_VOCAB_CHUNK = 4096
_TARGET_CHUNK = 16


def _row_blocks(scores: torch.Tensor):
    """Yield row views without flattening/copying noncontiguous leading axes."""
    if scores.ndim == 1:
        yield (), scores.unsqueeze(0)
        return
    for prefix in product(*(range(n) for n in scores.shape[:-2])):
        for start in range(0, scores.shape[-2], _ROW_CHUNK):
            index = (*prefix, slice(start, start + _ROW_CHUNK))
            yield index, scores[index]


def selected_token_ranks(
    scores: torch.Tensor, ids: torch.Tensor, *, row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """1-based ranks, descending score then ascending token ID, in bounded chunks.

    ``ids`` broadcasts to ``scores.shape[:-1] + [n_targets]``. For a 2D score
    matrix, optional ``row_indices`` selects rows *inside* vocabulary chunks,
    avoiding a full-vocabulary advanced-index copy at the caller. Exact ties
    never give more than k tokens a rank <= k. Nonfinite scores are unsupported.
    """
    ids = ids.to(device=scores.device, dtype=torch.long)
    if row_indices is None:
        shape = scores.shape[:-1]
        blocks = ((index, block, None) for index, block in _row_blocks(scores))
    else:
        if scores.ndim != 2 or row_indices.ndim != 1:
            raise ValueError("row_indices requires 2D scores and a 1D row index")
        row_indices = row_indices.to(device=scores.device, dtype=torch.long)
        shape = (len(row_indices),)
        blocks = (
            ((slice(start, start + _ROW_CHUNK),), scores,
             row_indices[start:start + _ROW_CHUNK])
            for start in range(0, len(row_indices), _ROW_CHUNK)
        )
    ids = ids.expand(*shape, ids.shape[-1])
    ranks = torch.ones(ids.shape, device=scores.device, dtype=torch.long)
    for index, block, rows in blocks:
        block_ids = ids[index].reshape(-1, ids.shape[-1]) if ids.shape[-1] else None
        if block_ids is None:
            continue
        for target in range(0, ids.shape[-1], _TARGET_CHUNK):
            selected = block_ids[:, target:target + _TARGET_CHUNK]
            values = (
                block.gather(-1, selected) if rows is None
                else block[rows[:, None], selected]
            )
            counts = torch.ones_like(selected)
            for start in range(0, scores.shape[-1], _VOCAB_CHUNK):
                chunk = block[:, start:start + _VOCAB_CHUNK]
                if rows is not None:
                    chunk = chunk[rows]
                vocab_ids = torch.arange(start, start + chunk.shape[-1], device=scores.device)
                greater = chunk[:, None, :] > values[..., None]
                tied = (chunk[:, None, :] == values[..., None]) & (vocab_ids < selected[..., None])
                counts += (greater | tied).sum(-1)
            ranks[index][..., target:target + _TARGET_CHUNK] = counts.reshape(
                ranks[index][..., target:target + _TARGET_CHUNK].shape
            )
    return ranks


def top_token_ids(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Compact lexical top-k; ties follow token ID, with bounded chunk sorting.

    Only chunk-local candidates and the running k winners are sorted/retained,
    never an all-position vocabulary permutation. For k >= half the vocabulary,
    a row-chunk permutation is already bounded by twice the requested output.
    The returned tensor owns
    exactly ``scores.shape[:-1] + (k,)`` storage, including k=0.
    """
    if not 0 <= k <= scores.shape[-1]:
        raise ValueError("k must be between zero and vocabulary size")
    result = torch.empty((*scores.shape[:-1], k), dtype=torch.long, device=scores.device)
    if k == 0:
        return result
    for index, block in _row_blocks(scores):
        if 2 * k >= scores.shape[-1]:
            # A large requested output already budgets this row-local permutation;
            # avoid repeatedly merging growing prefixes for k near vocabulary size.
            order = block.argsort(dim=-1, descending=True, stable=True)
            result[index] = order[:, :k].reshape(result[index].shape)
            del order
            continue
        winners = torch.empty((len(block), 0), dtype=torch.long, device=scores.device)
        for start in range(0, scores.shape[-1], _VOCAB_CHUNK):
            chunk = block[:, start:start + _VOCAB_CHUNK]
            local = chunk.argsort(dim=-1, descending=True, stable=True)[:, :k] + start
            candidates = torch.cat((winners, local), dim=-1)
            # Restore token-ID order before stable score sorting. Sorting at most
            # 2k candidates also repairs arbitrary ties at each chunk boundary.
            candidates = candidates.sort(dim=-1).values
            values = block.gather(-1, candidates)
            order = values.argsort(dim=-1, descending=True, stable=True)[:, :k]
            winners = candidates.gather(-1, order)
        result[index] = winners.reshape(result[index].shape)
    return result
