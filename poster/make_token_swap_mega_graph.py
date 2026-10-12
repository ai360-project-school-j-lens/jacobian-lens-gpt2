"""Combine the token-swap bootstrap results of three models in one figure.

Inputs (from `digitize_token_swap_figures.py`):
* `data/token_swap_digitized.csv` - per-band values read from the notebook charts;
* `data/token_swap_exact_primary.csv` - exact values for the primary (middle) band.

One panel per dataset; color = model, filled circle with solid interval = J-space
swap, hollow square with dashed interval = logit-direction swap; alpha = 1 at three relative-depth bands, with pointwise 95% bootstrap intervals.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
MODELS = {  # same validated slots as the latent-bridge mega-graph
    "gpt2-xl": ("GPT-2 XL", "#2a78d6"),
    "gemma-4-31b": ("Gemma 4 31B", "#eb6834"),
    "gemma-4-e4b": ("Gemma 4 E4B", "#1baf7a"),
}
ARMS = {"J": ("J-space swap", "-"), "logit": ("logit-direction swap", (0, (4, 2.5)))}
DATASETS = [("latent-bridge", "Latent bridge (country → language)"),
            ("probe-swap", "Probe-swap"),
            ("flexible-generalization", "Flexible generalization")]
BANDS = ["early", "middle", "late"]
BAND_LABELS = ["early\n20–40% depth", "middle (primary)\n40–60% depth", "late\n60–80% depth"]
OFFSETS = dict(zip(MODELS, (-0.14, 0.0, 0.14), strict=True))


def load() -> pd.DataFrame:
    digitized = pd.read_csv(HERE / "data/token_swap_digitized.csv").assign(source="digitized")
    exact = pd.read_csv(HERE / "data/token_swap_exact_primary.csv")
    exact = exact[exact.statistic.eq("arm")].assign(source="exact")
    keys = ["model", "dataset", "band", "arm"]
    columns = keys + ["estimate", "ci_low", "ci_high", "source"]
    data = pd.concat([digitized[columns], exact[columns]], ignore_index=True)
    return (data.sort_values("source").drop_duplicates(keys, keep="last")
            .sort_values(keys).reset_index(drop=True))


def plot(data: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS, strict=True):
        for model, (_, color) in MODELS.items():
            for arm, (_, style) in ARMS.items():
                rows = (data[data.model.eq(model) & data.dataset.eq(dataset) & data.arm.eq(arm)]
                        .set_index("band").reindex(BANDS))
                x = np.arange(len(BANDS)) + OFFSETS[model] + (0.05 if arm == "logit" else 0)
                y = rows.estimate.to_numpy(float)
                low = (y - rows.ci_low.to_numpy(float)).clip(min=0)
                high = (rows.ci_high.to_numpy(float) - y).clip(min=0)
                # Points and intervals only: J filled circle with a solid interval,
                # logit hollow square with a dashed interval.
                markers = (dict(marker="o", markerfacecolor=color, markeredgecolor="white",
                                markeredgewidth=0.9, markersize=8) if arm == "J" else
                           dict(marker="s", markerfacecolor="white", markeredgecolor=color,
                                markeredgewidth=1.8, markersize=7))
                bars = ax.errorbar(x, y, yerr=np.nan_to_num(np.vstack([low, high])), color=color,
                                   linestyle="none", elinewidth=1.6, capsize=3.5, capthick=1.6,
                                   zorder=3, **markers)
                bars[2][0].set_linestyle(style)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(range(len(BANDS)), BAND_LABELS)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.02, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8a8984")
        ax.tick_params(colors="#52514e")
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Target flips / correct-baseline prompts")

    model_handles = [Line2D([], [], color=c, linewidth=3, label=name) for name, c in MODELS.values()]
    arm_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor="#3d3c39",
               markeredgecolor="white", label=f"{ARMS['J'][0]} (solid interval)"),
        Line2D([], [], linestyle="none", marker="s", markersize=8, markerfacecolor="white",
               markeredgecolor="#3d3c39", markeredgewidth=1.8,
               label=f"{ARMS['logit'][0]} (dashed interval)"),
    ]
    fig.legend(handles=model_handles + arm_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.945), frameon=False, fontsize=13, handlelength=3)
    fig.suptitle("Token swaps: concept transfer by J-space vs logit-direction swaps across models",
                 x=0.5, y=0.995, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.008,
             "α = 1; markers = success rate among prompts with a correct baseline, vertical bars = pointwise "
             "95% prompt-clustered bootstrap intervals (10,000 resamples; none where degenerate).\n"
             "Norm-matched random swaps stay at or below 2% in every band (not shown). Primary band "
             "exact; other bands read from the notebooks' charts (error at most 0.01).",
             ha="center", va="bottom", fontsize=11, color="#52514e", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 0.915))
    return fig


def main() -> None:
    data = load()
    counts = data.groupby(["model", "dataset", "arm"]).size()
    assert (counts == len(BANDS)).all(), counts[counts != len(BANDS)]
    data.to_csv(HERE / "data/token_swap_mega_graph_data.csv", index=False)
    fig = plot(data)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"token_swap_bootstrap_mega.{suffix}", dpi=300)
    print("saved token_swap_bootstrap_mega.{png,svg,pdf} and data/token_swap_mega_graph_data.csv;",
          len(data), "values,", int(data.source.eq("exact").sum()), "exact")


if __name__ == "__main__":
    main()
