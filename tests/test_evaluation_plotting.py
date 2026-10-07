"""Regression tests for dataset-panel plotting, without model evaluation."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from jlens.evaluation import DATASETS, plot_layer_curves


@pytest.mark.parametrize("logy", [False, True])
def test_series_colors_match_legend_when_targets_are_absent(logy):
    plt.switch_backend("Agg")
    curves = {}
    for lens in ("logit lens", "J-lens"):
        for kind in ("intermediate", "control", "target"):
            datasets = DATASETS[:3] if kind == "target" else DATASETS
            curves[f"{lens}: {kind}"] = pd.DataFrame(
                [[1, 2, 3]] * len(datasets), index=datasets
            )

    # Respect the active palette as well as keeping its assignments consistent.
    palette = ["navy", "orange", "green", "red", "purple", "brown"]
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=palette)}), patch.object(plt, "show"):
        try:
            plot_layer_curves(curves, "Test curves", "Metric", logy=logy)
            fig = plt.gcf()
            expected_colors = dict(zip(curves, palette, strict=True))
            legend = fig.axes[0].get_legend()
            legend_colors = {
                text.get_text(): line.get_color()
                for text, line in zip(legend.get_texts(), legend.get_lines(), strict=True)
            }
            assert legend_colors == expected_colors
            assert len(fig.axes) == len(DATASETS)
            for ax, dataset in zip(fig.axes, DATASETS, strict=True):
                expected_labels = [
                    label for label, frame in curves.items() if dataset in frame.index
                ]
                assert ax.get_title() == dataset
                assert [line.get_label() for line in ax.lines] == expected_labels
                assert ax.get_yscale() == ("log" if logy else "linear")
                for line in ax.lines:
                    label = line.get_label()
                    assert line.get_color() == legend_colors[label]
                    assert line.get_linestyle() == ("--" if "control" in label else "-")
                if dataset in ("poetry", "association", "typo"):
                    assert len(ax.lines) == 4
                    assert not any("target" in label for label in expected_labels)
        finally:
            plt.close("all")
