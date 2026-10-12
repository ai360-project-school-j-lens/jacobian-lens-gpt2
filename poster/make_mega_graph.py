"""Combine the bootstrapped latent-bridge pass@k curves of three models in one figure.

Inputs (from `extract_exact_k10.py` and `digitize_bootstrap_figures.py`):
* `data/digitized_pass_at_k.csv` - curves read back from the notebook figures;
* `data/exact_pass_at_10.csv` - exact pass@10 values and intervals, which replace
  the digitized values at k = 10.

One panel per dataset shared by all three runs; color = model, solid = J-lens,
dashed = logit lens. Intermediates only, with pointwise 95% bootstrap bands.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
MODELS = {  # validated categorical slots 1-3 (dataviz reference palette, all pairs)
    "gpt2-xl": ("GPT-2 XL", "#2a78d6"),
    "gemma-4-31b": ("Gemma 4 31B", "#eb6834"),
    "gemma-4-e4b": ("Gemma 4 E4B", "#1baf7a"),
}
LENSES = {"J-lens": "-", "logit lens": (0, (4, 2.5))}
PANELS = [  # (dataset, title)
    ("latent bridge: all but slot", "Latent bridge (all but slot)"),
    ("multihop", "Multihop"),
    ("multihop-easy", "Multihop-easy"),
    ("typo", "Typo"),
    ("multilingual", "Multilingual"),
    ("association", "Association"),
    ("order-ops", "Order-ops"),
    ("poetry", "Poetry"),
]
KS = [1, 2, 5, 10, 20, 50, 100]


def load_curves() -> pd.DataFrame:
    digitized = pd.read_csv(HERE / "data/digitized_pass_at_k.csv").assign(source="digitized")
    exact = pd.read_csv(HERE / "data/exact_pass_at_10.csv")
    exact = exact[exact.statistic.eq("arm")].assign(source="exact")
    keys = ["model", "dataset", "arm", "kind", "k"]
    columns = keys + ["estimate", "ci_low", "ci_high", "source"]
    curves = pd.concat([digitized[columns], exact[columns]], ignore_index=True)
    # Prefer exact rows where both exist (k = 10).
    curves = curves.sort_values("source").drop_duplicates(keys, keep="last")
    curves = curves[curves.kind.eq("intermediate") & curves.dataset.isin([d for d, _ in PANELS])]
    return curves.sort_values(keys).reset_index(drop=True)


def plot(curves: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), sharex=True, sharey=True)
    for ax, (dataset, title) in zip(axes.flat, PANELS, strict=True):
        for model, (_, color) in MODELS.items():
            for arm, style in LENSES.items():
                line = curves[curves.model.eq(model) & curves.dataset.eq(dataset)
                              & curves.arm.eq(arm)].sort_values("k")
                ks = line.k.to_numpy(float)
                ax.fill_between(ks, line.ci_low.to_numpy(float), line.ci_high.to_numpy(float),
                                color=color, alpha=0.09, linewidth=0)
                ax.plot(ks, line.estimate, color=color, linestyle=style, linewidth=2.2,
                        marker="o", markersize=4.5, markeredgecolor="white",
                        markeredgewidth=0.8, solid_capstyle="round", dash_capstyle="round")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.minorticks_off()
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8a8984")
        ax.tick_params(colors="#52514e")
    for ax in axes[1]:
        ax.set_xlabel("k (rank cutoff)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Inner-layer pass@k")

    model_handles = [Line2D([], [], color=c, linewidth=3, label=name) for name, c in MODELS.values()]
    lens_handles = [Line2D([], [], color="#3d3c39", linewidth=2.2, linestyle=s, label=arm)
                    for arm, s in LENSES.items()]
    fig.legend(handles=model_handles + lens_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=13, handlelength=3)
    fig.suptitle("Annotated-intermediate retrieval: J-lens vs logit lens across models",
                 x=0.5, y=0.995, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.008,
             "Inner layers; lines = point estimates, shaded = pointwise 95% paired-bootstrap "
             "intervals (10,000 prompt-cluster resamples; none drawn where degenerate).\n"
             "Values read from the benchmark notebooks' figures, exact at k = 10; "
             "digitizing error at k = 10 is at most 0.006 for estimates and 0.03 for interval bounds.",
             ha="center", va="bottom", fontsize=11, color="#52514e", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    return fig


def main() -> None:
    curves = load_curves()
    missing = curves.groupby(["model", "dataset", "arm"]).size()
    assert (missing == len(KS)).all(), missing[missing != len(KS)]
    curves.to_csv(HERE / "data/mega_graph_data.csv", index=False)
    fig = plot(curves)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"latent_bridge_bootstrap_mega.{suffix}", dpi=300)
    print("saved", ", ".join(f"latent_bridge_bootstrap_mega.{s}" for s in ("png", "svg", "pdf")),
          "and data/mega_graph_data.csv;", len(curves), "points,",
          int(curves.source.eq("exact").sum()), "exact")


if __name__ == "__main__":
    main()
