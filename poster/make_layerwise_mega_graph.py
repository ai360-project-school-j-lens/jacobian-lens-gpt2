"""Combine per-layer latent-bridge pass@10 readouts of three models in one figure.

Inputs are the exact exports of the `..._bootstrap_layerwise` notebooks:
* `data/latent_bridge_layer_profiles_gpt2-xl.csv` (GPT-2 XL, latent-bridge rows);
* `data/layerwise/gemma-4-31b-layerwise_plot_data.csv`;
* `data/layerwise/gemma-4-e4b_layerwise_plot_data.csv`.

Each score uses one block (no minimum over layers); latent-bridge ranks take the
minimum over positions in the span. One panel per span; x is normalized depth
(block / (blocks - 1)), color = model, solid = J-lens, dashed = logit lens;
shaded = pointwise 95% paired-bootstrap intervals where one exists.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCES = {
    "gpt2-xl": "data/latent_bridge_layer_profiles_gpt2-xl.csv",
    "gemma-4-31b": "data/layerwise/gemma-4-31b-layerwise_plot_data.csv",
    "gemma-4-e4b": "data/layerwise/gemma-4-e4b_layerwise_plot_data.csv",
}
MODELS = {  # same validated slots as the other merged graphs
    "gpt2-xl": ("GPT-2 XL (48 blocks)", "#2a78d6"),
    "gemma-4-31b": ("Gemma 4 31B (60 blocks)", "#eb6834"),
    "gemma-4-e4b": ("Gemma 4 E4B (42 blocks)", "#1baf7a"),
}
LENSES = {"J-lens": "-", "logit lens": (0, (4, 2.5))}
SPANS = [
    ("latent bridge: all but slot", "All positions but the slot"),
    ("latent bridge: country token", "Country token"),
    ("latent bridge: subject", "Subject tokens"),
]
COLUMNS = ["model", "dataset", "kind", "statistic", "arm", "layer", "n_layers",
           "depth_percent", "estimate", "ci_low", "ci_high", "status"]


def load() -> pd.DataFrame:
    frames = []
    for model, path in SOURCES.items():
        frame = pd.read_csv(HERE / path, low_memory=False).assign(model=model)
        if "metric" in frame:
            frame = frame[frame.metric.eq("pass_at_k") & frame.split.eq("all")]
        frame = frame[frame.k.eq(10) & frame.dataset.isin([d for d, _ in SPANS])]
        frames.append(frame[COLUMNS])
    data = pd.concat(frames, ignore_index=True)
    return data[data.kind.eq("intermediate") & data.statistic.eq("arm")].reset_index(drop=True)


def plot(data: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
    for ax, (dataset, title) in zip(axes, SPANS, strict=True):
        for model, (_, color) in MODELS.items():
            for arm, style in LENSES.items():
                line = data[data.model.eq(model) & data.dataset.eq(dataset)
                            & data.arm.eq(arm)].sort_values("layer")
                x = line.depth_percent.to_numpy(float)
                ok = line.status.eq("ok")
                ax.fill_between(x, line.ci_low.where(ok).to_numpy(float),
                                line.ci_high.where(ok).to_numpy(float),
                                color=color, alpha=0.10, linewidth=0)
                ax.plot(x, line.estimate, color=color, linestyle=style, linewidth=2.2,
                        solid_capstyle="round", dash_capstyle="round")
        ax.axvline(100, color="#8a8984", linestyle=":", linewidth=1)
        ax.set_title(f"Latent bridge — {title}", loc="left", fontweight="bold")
        ax.set_xlim(-1, 101)
        ax.set_xticks(np.linspace(0, 100, 6))
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xlabel("Block depth (% of network)")
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#8a8984")
        ax.tick_params(colors="#52514e")
    axes[0].set_ylabel("Per-layer pass@10")

    model_handles = [Line2D([], [], color=c, linewidth=3, label=name) for name, c in MODELS.values()]
    lens_handles = [Line2D([], [], color="#3d3c39", linewidth=2.2, linestyle=s, label=arm)
                    for arm, s in LENSES.items()]
    fig.legend(handles=model_handles + lens_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 0.945), frameon=False, fontsize=13, handlelength=3)
    fig.suptitle("Latent bridge: where the bridging country is readable, layer by layer",
                 x=0.5, y=0.995, fontsize=17, fontweight="bold")
    fig.text(0.5, 0.008,
             "Pass@10 at a single block (country word in the top 10, minimum over the span's "
             "positions); shaded = pointwise 95% paired-bootstrap intervals (10,000 resamples), "
             "not corrected across blocks.\nDotted line = final block, shared by both lenses. "
             "Exact values from each model's layerwise notebook export; 530 latent-bridge prompts.",
             ha="center", va="bottom", fontsize=11, color="#52514e", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 0.915))
    return fig


def main() -> None:
    data = load()
    counts = data.groupby(["model", "dataset", "arm"]).size()
    expected = data.groupby(["model", "dataset", "arm"]).n_layers.first()
    assert (counts == expected).all(), counts[counts != expected]
    data.to_csv(HERE / "data/layerwise_mega_graph_data.csv", index=False)
    fig = plot(data)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"latent_bridge_layerwise_mega.{suffix}", dpi=300)
    print("saved latent_bridge_layerwise_mega.{png,svg,pdf} and data/layerwise_mega_graph_data.csv;",
          len(data), "points")
    print(data.groupby(["model", "dataset", "arm"]).estimate.agg(["max", "idxmax"]).assign(
        peak_depth=lambda t: data.loc[t["idxmax"], "depth_percent"].round(0).to_numpy())
          .drop(columns="idxmax").round(3).to_string())


if __name__ == "__main__":
    main()
