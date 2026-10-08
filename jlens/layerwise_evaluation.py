"""Compatibility entry point for the shared streaming paired evaluator."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from jlens.evaluation import evaluate_paired
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.readout import LensReadout


def evaluate_paired_layerwise(
    model: LensModel,
    lens: JacobianLens,
    evals: dict[str, list[dict]],
    desc: str = "paired lenses (layer-bounded)",
    *,
    logit_readout: Callable[..., LensReadout] | None = None,
    spelling_lookup: Callable[[str, bool], set[int]] | None = None,
    preserve_prompt_whitespace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delegate to :func:`jlens.evaluation.evaluate_paired` without restacking.

    Preserve this helper's callback contract: ``logit_readout`` accepts
    ``layers=[layer]`` and returns a dual readout of shape [1, sequence, vocab].
    The shared evaluator exposes that option as ``layer_logit_readout``; its
    original ``logit_readout`` option still accepts full tensors/dual readouts.

    Both entry points stream by default, retaining the final reference and one
    current readout. All block activations remain live for the current prompt;
    distribution workspace scales with sequence length and vocabulary, not depth.
    Scoring, whitespace, table ordering and behavior metrics are identical.
    """
    return evaluate_paired(
        model, lens, evals, desc=desc,
        layer_logit_readout=logit_readout,
        spelling_lookup=spelling_lookup,
        preserve_prompt_whitespace=preserve_prompt_whitespace,
    )
