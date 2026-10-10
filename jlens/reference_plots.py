# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Reference-style figure adaptations from cached evaluation/metrics tables.

No inference, downloads, or fitting occurs here. See docs/reference_plots.md
for aggregation conventions and differences from the reference experiments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

PANEL_DATASETS = (
    "multihop", "multilingual", "poetry", "order-ops", "association", "typo"
)
SWEEP_KS = (1, 2, 5, 10, 20, 50, 100)
LENS_NAMES = ("J-lens", "logit lens")


@dataclass(frozen=True)
class IntermediateSweep:
    """Item-weighted scores/AUC and single-token exclusion counts per lens/task."""

    scores: pd.DataFrame
    counts: pd.DataFrame


def intermediate_rank_sweep(
    words: pd.DataFrame, *, ks: Sequence[int] = SWEEP_KS
) -> IntermediateSweep:
    """Figure 52 adaptation: intermediate-only, single-token, best over layers.

    Average retained word hits within each item, then equally across items.
    Items without retained words are excluded, not assigned zero. Counts are
    per lens (the paired evaluation repeats annotations for each lens).
    AUC is trapezoidal in log(k), divided by log(k_max/k_min).
    """
    k_values = np.asarray(ks, dtype=float)
    if (
        k_values.ndim != 1 or len(k_values) < 2
        or not np.isfinite(k_values).all() or (k_values < 1).any()
        or (k_values != np.floor(k_values)).any()
        or (np.diff(k_values) <= 0).any()
    ):
        raise ValueError("ks must be at least two strictly increasing positive integers")
    rows, counts = [], []
    intermediate = words[
        words["kind"].eq("intermediate") & words["lens"].isin(LENS_NAMES)
    ]
    for (dataset, name), group in intermediate.groupby(["dataset", "lens"]):
        retained = group[group["single_token"].eq(True)].copy()
        counts.append({
            "dataset": dataset, "lens": name,
            "n_words_total": len(group), "n_words_used": len(retained),
            "n_words_excluded": len(group) - len(retained),
            "n_items_total": group["item"].nunique(),
            "n_items_used": retained["item"].nunique(),
            "n_items_excluded": group["item"].nunique() - retained["item"].nunique(),
        })
        if retained.empty:
            scores = np.full(len(k_values), np.nan)
        else:
            ranks = np.stack(retained["ranks"])
            if (
                ranks.ndim != 2 or ranks.shape[1] == 0
                or not np.isfinite(ranks).all() or (ranks < 1).any()
            ):
                raise ValueError("ranks must be finite, positive per-layer arrays")
            best = ranks.min(axis=1)
            scores = np.asarray([
                retained.assign(_hit=best <= k).groupby("item")["_hit"].mean().mean()
                for k in k_values
            ])
        log_k = np.log(k_values)
        # Explicit trapezoids support both NumPy 1.x and 2.x.
        auc = np.sum(np.diff(log_k) * (scores[:-1] + scores[1:]) / 2)
        auc /= log_k[-1] - log_k[0]
        rows.extend({
            "dataset": dataset, "lens": name, "k": int(k),
            "score": float(score), "auc": float(auc),
        } for k, score in zip(k_values, scores, strict=True))
    return IntermediateSweep(
        pd.DataFrame(rows, columns=["dataset", "lens", "k", "score", "auc"]),
        pd.DataFrame(counts, columns=[
            "dataset", "lens", "n_words_total", "n_words_used", "n_words_excluded",
            "n_items_total", "n_items_used", "n_items_excluded",
        ]),
    )


def plot_intermediate_sweep(sweep: IntermediateSweep) -> tuple[Figure, np.ndarray]:
    """Plot the 2x3 Figure 52 adaptation; return figure/axes without showing."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    colors = {"J-lens": "black", "logit lens": "red"}
    for ax, dataset in zip(axes.flat, PANEL_DATASETS, strict=True):
        for name in LENS_NAMES:
            frame = sweep.scores[
                sweep.scores["dataset"].eq(dataset) & sweep.scores["lens"].eq(name)
            ].sort_values("k")
            if frame.empty or not frame["score"].notna().any():
                continue
            ax.plot(
                frame["k"], frame["score"], color=colors[name], marker="o",
                label=f"{name} (AUC {frame['auc'].iloc[0]:.2f})",
            )
        ax.set(title=dataset, xscale="log", ylim=(0, 1),
               xlabel="k", ylabel="Intermediate hit rate")
        ticks = sorted(sweep.scores["k"].unique()) or list(SWEEP_KS)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.minorticks_off()
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, "No eligible intermediates", ha="center",
                    transform=ax.transAxes)
    fig.suptitle("Figure 52 adaptation — single-token intermediates, best over all layers")
    fig.tight_layout()
    return fig, axes


def plot_intermediate_auc(
    sweep: IntermediateSweep, *, datasets: Sequence[str] = PANEL_DATASETS,
) -> tuple[Figure, Axes]:
    """Compare the sweep's normalized log-k AUC by dataset; missing stays N/A."""
    auc = sweep.scores[["dataset", "lens", "auc"]].drop_duplicates()
    values = auc.pivot(index="dataset", columns="lens", values="auc").reindex(
        index=list(datasets), columns=list(LENS_NAMES),
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(values))
    width = 0.38
    for index, (name, color) in enumerate(zip(LENS_NAMES, ("black", "red"), strict=True)):
        positions = x + (index - 0.5) * width
        ax.bar(positions, values[name], width, label=name, color=color)
        for position, value in zip(positions, values[name], strict=True):
            ax.text(position, 0.02 if pd.isna(value) else value + 0.015,
                    "N/A" if pd.isna(value) else f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ks = sorted(sweep.scores["k"].unique())
    interval = f" (k={ks[0]}–{ks[-1]})" if ks else ""
    ax.set_xticks(x, values.index, rotation=25, ha="right")
    ax.set(xlim=(-0.6, max(len(values), 1) - 0.4), ylim=(0, 1.08),
           ylabel="Normalized AUC over log(k)",
           title="Intermediate pass@k AUC by dataset" + interval
           + "\nBest over all layers, including shared model output")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def normalized_depth(layers: Sequence[int], n_layers: int) -> np.ndarray:
    """Map actual block indices to 0..100; never rescale a selected subset.

    First block output is 0, final block output is 100 (no embedding row).
    A one-block model has only its final output, at 100.
    """
    indices = np.asarray(layers, dtype=float)
    if not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError("n_layers must be a positive integer")
    if (
        indices.ndim != 1 or not np.isfinite(indices).all()
        or (indices != np.floor(indices)).any()
        or (indices < 0).any() or (indices >= n_layers).any()
    ):
        raise ValueError("layers must be integer block indices in the model range")
    return indices * 100 / (n_layers - 1) if n_layers > 1 else indices * 0 + 100


def same_layer_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select J/logit comparisons only, accepting either pair orientation.

    Supports both same-layer-only and legacy all-pairs metrics tables. Never
    average cross-layer pairs or silently double-count mirrored pairs.
    """
    cross_lens = (
        (pairs["lens_a"].eq("J-lens") & pairs["lens_b"].eq("logit lens"))
        | (pairs["lens_a"].eq("logit lens") & pairs["lens_b"].eq("J-lens"))
    )
    frame = pairs[cross_lens & pairs["layer_a"].eq(pairs["layer_b"])].copy()
    frame["layer"] = frame["layer_a"]
    if frame["layer"].duplicated().any():
        raise ValueError("Expected exactly one J/logit pair per layer, not mirrored pairs")
    return frame.sort_values("layer").reset_index(drop=True)


def _depth_axes(axes: np.ndarray) -> None:
    for ax in axes:
        ax.set(xlim=(0, 100), xlabel="Normalized block depth (%)")
        ax.grid(alpha=0.2)


def _empty_panel(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
            transform=ax.transAxes)


def plot_distribution_summary(
    layers: pd.DataFrame, *, n_layers: int
) -> tuple[Figure, np.ndarray]:
    """Figure 55 adaptation: token-weighted KL, entropy, model agreement (%)."""
    frame = layers[layers["lens"].isin(LENS_NAMES)]
    if frame.duplicated(["lens", "layer"]).any():
        raise ValueError("Expected one metrics row per lens/layer; select one corpus")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"J-lens": "blue", "logit lens": "purple"}
    specs = (
        ("kl_model_to_lens", "KL(model || lens)", "KL (nats)", 1),
        ("entropy", "Lens entropy", "Entropy (nats)", 1),
        ("top1_agreement", "Top-1 agreement with model", "Agreement (%)", 100),
    )
    for ax, (metric, title, ylabel, scale) in zip(axes, specs, strict=True):
        for name in LENS_NAMES:
            group = frame[frame["lens"].eq(name)].sort_values("layer")
            if not group.empty:
                ax.plot(normalized_depth(group["layer"], n_layers),
                        group[metric] * scale, color=colors[name], label=name)
        ax.set(title=title, ylabel=ylabel)
        if ax.lines:
            ax.legend()
        else:
            _empty_panel(ax, "No distribution metrics")
    axes[2].set_ylim(0, 100)
    _depth_axes(axes)
    fig.suptitle("Figure 55 adaptation — held-out distribution diagnostics")
    fig.tight_layout()
    return fig, axes


def plot_lens_comparison(
    pairs: pd.DataFrame, geometry: pd.DataFrame, *, n_layers: int
) -> tuple[Figure, np.ndarray]:
    """Figure 56 adaptation: linear-head cosine and same-layer J/logit metrics.

    symmetric_kl is already the half-sum in jlens.metrics; do not halve again.
    Unsupported/zero-vector-only geometry is shown as unavailable, never zero.
    """
    frame = same_layer_pairs(pairs)
    if geometry["layer"].duplicated().any():
        raise ValueError("Expected one geometry row per layer")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    supported = geometry.sort_values("layer").copy()
    # Keep missing/unsupported rows as NaN so the line does not bridge them.
    supported.loc[supported["status"].ne("ok"), "mean_cosine"] = np.nan
    if supported["mean_cosine"].notna().any():
        axes[0].plot(normalized_depth(supported["layer"], n_layers),
                     supported["mean_cosine"], color="purple", label="J-lens / logit lens")
        axes[0].legend()
    else:
        _empty_panel(axes[0], "Linear-head geometry unavailable\nSee geometry status/reason")
    has_negative_cosine = supported["mean_cosine"].lt(0).any()
    cosine_title = "Vocabulary-vector cosine (linear head only)"
    if has_negative_cosine:
        cosine_title += "\nAxis extended below reference range to show negatives"
    axes[0].set(title=cosine_title, ylabel="Mean cosine",
                ylim=(-1 if has_negative_cosine else 0, 1))
    for ax, metric, title, ylabel in (
        (axes[1], "symmetric_kl", "Same-layer symmetric KL (half-sum)", "KL (nats)"),
        (axes[2], "top1_agreement", "Same-layer top-1 agreement", "Agreement"),
    ):
        if not frame.empty:
            ax.plot(normalized_depth(frame["layer"], n_layers), frame[metric],
                    color="purple", label="J-lens / logit lens")
            ax.legend()
        else:
            _empty_panel(ax, "No same-layer J/logit pairs")
        ax.set(title=title, ylabel=ylabel)
    axes[2].set_ylim(0, 1)
    _depth_axes(axes)
    fig.suptitle("Figure 56 adaptation — J-lens versus logit lens")
    fig.tight_layout()
    return fig, axes
