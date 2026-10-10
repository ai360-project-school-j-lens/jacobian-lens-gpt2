# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Offline aggregation and figure-layout regressions (no model downloads)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jlens.reference_plots import (
    PANEL_DATASETS,
    SWEEP_KS,
    intermediate_rank_sweep,
    normalized_depth,
    plot_distribution_summary,
    plot_intermediate_auc,
    plot_intermediate_sweep,
    plot_lens_comparison,
    same_layer_pairs,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def word_frame():
    rows = []
    for dataset in PANEL_DATASETS:
        for name in ("J-lens", "logit lens"):
            # Item 0: one hit among three words; item 1: one hit. Score is
            # (1/3 + 1)/2 = 2/3, NOT the word-weighted 2/4. Best rank includes
            # final output: the first word only hits there.
            for item, ranks, single, kind in (
                (0, [200, 1], True, "intermediate"),
                (0, [200, 200], True, "intermediate"),
                (0, [200, 200], True, "intermediate"),
                (1, [1, 200], True, "intermediate"),
                (1, [200, 200], False, "intermediate"),
                (2, [1, 1], False, "intermediate"),
                (3, [200, 200], True, "target"),
                (3, [200, 200], True, "control"),
            ):
                rows.append(dict(dataset=dataset, lens=name, item=item,
                                 ranks=ranks, single_token=single, kind=kind))
    return pd.DataFrame(rows)


def test_sweep_item_weighting_filter_counts_auc_and_no_mutation():
    words = word_frame()
    original = words.copy(deep=True)
    sweep = intermediate_rank_sweep(words)
    np.testing.assert_allclose(sweep.scores.score, 2 / 3)
    np.testing.assert_allclose(sweep.scores.auc, 2 / 3)
    assert sweep.scores.k.unique().tolist() == list(SWEEP_KS)
    assert (sweep.counts.n_words_total == 6).all()
    assert (sweep.counts.n_words_used == 4).all()
    assert (sweep.counts.n_words_excluded == 2).all()
    assert (sweep.counts.n_items_total == 3).all()
    assert (sweep.counts.n_items_used == 2).all()
    assert (sweep.counts.n_items_excluded == 1).all()
    pd.testing.assert_frame_equal(words, original)


def test_auc_integrates_log_k_not_linear_k():
    words = word_frame().iloc[:1].copy()
    words.at[0, "ranks"] = [200, 10]
    sweep = intermediate_rank_sweep(words, ks=[1, 10, 100])
    assert sweep.scores.score.tolist() == [0, 1, 1]
    np.testing.assert_allclose(sweep.scores.auc, 0.75)


def test_auc_bars_match_sweep_and_keep_missing_datasets_unavailable():
    words = word_frame()
    words.loc[words.lens.eq("logit lens"), "single_token"] = False
    sweep = intermediate_rank_sweep(words)
    before = sweep.scores.copy(deep=True)
    datasets = ["order-ops", "multihop", "absent"]
    _, ax = plot_intermediate_auc(sweep, datasets=datasets)
    np.testing.assert_allclose(
        [bar.get_height() for bar in ax.containers[0]], [2 / 3, 2 / 3, np.nan],
    )
    assert all(np.isnan(bar.get_height()) for bar in ax.containers[1])
    assert [label.get_text() for label in ax.get_xticklabels()] == datasets
    assert sum(text.get_text() == "N/A" for text in ax.texts) == 4
    pd.testing.assert_frame_equal(sweep.scores, before)
    _, empty_ax = plot_intermediate_auc(intermediate_rank_sweep(words.iloc[:0]))
    assert all(text.get_text() == "N/A" for text in empty_ax.texts)


@pytest.mark.parametrize("ks", [[1], [1, 1], [2, 1], [0, 1], [1, 1.5], [1, np.nan]])
def test_invalid_sweep_ks(ks):
    with pytest.raises(ValueError, match="ks"):
        intermediate_rank_sweep(word_frame(), ks=ks)


def test_figure52_layout_legends_colors_and_absent_data():
    sweep = intermediate_rank_sweep(word_frame())
    fig, axes = plot_intermediate_sweep(sweep)
    assert axes.shape == (2, 3)
    assert len(fig.axes) == 6
    for dataset, ax in zip(PANEL_DATASETS, axes.flat, strict=True):
        assert ax.get_title() == dataset
        assert ax.get_xscale() == "log"
        assert ax.get_ylim() == (0, 1)
        assert [line.get_color() for line in ax.lines] == ["black", "red"]
        assert [text.get_text() for text in ax.get_legend().texts] == [
            "J-lens (AUC 0.67)", "logit lens (AUC 0.67)",
        ]
        np.testing.assert_array_equal(ax.get_xticks(), SWEEP_KS)
    words = word_frame()
    words["single_token"] = False
    empty = intermediate_rank_sweep(words)
    assert empty.scores.score.isna().all()  # Not a fabricated zero hit rate.
    _, axes = plot_intermediate_sweep(empty)
    assert all(not ax.lines and ax.texts for ax in axes.flat)
    _, axes = plot_intermediate_sweep(intermediate_rank_sweep(words.iloc[:0]))
    assert all(not ax.lines for ax in axes.flat)


def test_normalized_depth_uses_model_not_subset_and_handles_one_block():
    np.testing.assert_allclose(normalized_depth([1, 3], 5), [25, 75])
    np.testing.assert_allclose(normalized_depth([0, 4], 5), [0, 100])
    np.testing.assert_allclose(normalized_depth([0], 1), [100])
    with pytest.raises(ValueError):
        normalized_depth([5], 5)
    with pytest.raises(ValueError):
        normalized_depth([0.5], 5)


def distribution_frame():
    return pd.DataFrame([
        dict(lens=name, layer=layer, kl_model_to_lens=0.2,
             entropy=2.0, top1_agreement=0.25)
        for name in ("J-lens", "logit lens") for layer in (1, 3)
    ])


def test_figure55_directions_percent_colors_and_actual_depth():
    layers = distribution_frame()
    fig, axes = plot_distribution_summary(layers, n_layers=5)
    assert len(fig.axes) == 3
    assert axes.shape == (3,)
    assert axes[0].get_title() == "KL(model || lens)"
    for ax, value in zip(axes, (0.2, 2.0, 25), strict=True):
        assert [line.get_color() for line in ax.lines] == ["blue", "purple"]
        for line in ax.lines:
            np.testing.assert_allclose(line.get_xdata(), [25, 75])
            np.testing.assert_allclose(line.get_ydata(), value)
        assert ax.get_xlim() == (0, 100)
        assert not ax.patches  # No reference-model workspace shading.
    assert axes[2].get_ylim() == (0, 100)
    with pytest.raises(ValueError, match="one metrics row"):
        plot_distribution_summary(pd.concat([layers, layers]), n_layers=5)


def pair_frame():
    return pd.DataFrame([
        dict(lens_a=a, layer_a=i, lens_b=b, layer_b=j,
             symmetric_kl=value, top1_agreement=0.6)
        for a, i, b, j, value in (
            ("logit lens", 0, "J-lens", 0, 0.2),
            ("J-lens", 2, "logit lens", 2, 0.4),  # Either orientation accepted.
            ("logit lens", 0, "J-lens", 2, 100),  # Exclude cross-layer pair.
            ("J-lens", 0, "J-lens", 0, 0),  # Exclude within-lens diagonal.
        )
    ])


def test_figure56_same_layer_pairs_half_sum_not_halved_again_and_geometry():
    pairs = pair_frame()
    selected = same_layer_pairs(pairs)
    assert selected.layer.tolist() == [0, 2]
    assert selected.symmetric_kl.tolist() == [0.2, 0.4]
    geometry = pd.DataFrame(dict(layer=[0, 2], mean_cosine=[0.1, 0.9], status="ok"))
    fig, axes = plot_lens_comparison(pairs, geometry, n_layers=3)
    assert len(fig.axes) == 3
    for ax, expected in zip(axes, ([0.1, 0.9], [0.2, 0.4], [0.6, 0.6]), strict=True):
        assert len(ax.lines) == 1
        assert ax.lines[0].get_color() == "purple"
        np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, 100])
    assert axes[0].get_ylim() == (0, 1)
    assert axes[2].get_ylim() == (0, 1)
    geometry.loc[0, "mean_cosine"] = -0.3
    _, axes = plot_lens_comparison(pairs, geometry, n_layers=3)
    assert axes[0].get_ylim() == (-1, 1)
    assert "extended below reference range" in axes[0].get_title()
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [-0.3, 0.9])
    geometry["status"] = "unsupported"
    geometry["mean_cosine"] = np.nan
    _, axes = plot_lens_comparison(pairs.iloc[:0], geometry, n_layers=3)
    assert all(not ax.lines and ax.texts for ax in axes)
    assert axes[0].get_ylim() == (0, 1)
    with pytest.raises(ValueError, match="one J/logit pair"):
        same_layer_pairs(pd.concat([pairs, pairs.iloc[:1]]))
