"""Convergence summary and plots from ``curves.py`` outputs.

Two sources of uncertainty, both for the J-lens readout at N=134 fit prompts:
  fit   -- which prompts the lens was fitted on: source-stratified bootstrap of
           the fit prompts, and disjoint 67/67 halves (sd at 134 ~ sd(h1 - h2) / 2)
  eval  -- which items the eval set contains: item bootstrap within each dataset
The learning curve asks whether the readout still moves near N=134: mean
|metric(N) - metric(134)| over random prompt orders.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cilib import load_eval_cache

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
parser.add_argument("--suffix", default="")
parser.add_argument("--item-boot", type=int, default=2000)
args = parser.parse_args()
run = args.run_dir
out = run / f"report{args.suffix}"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(1)

curves = pd.read_parquet(run / f"curves{args.suffix}.parquet")
words = load_eval_cache(run / "eval_cache.pt")["words"]
R_snap = np.load(run / f"ranks_snapshot{args.suffix}.npy")
R_logit = np.load(run / f"ranks_logit{args.suffix}.npy")
N_FULL = int(curves.loc[curves.experiment == "order", "N"].max())
DATASETS = list(dict.fromkeys(words.dataset))
J_COLOR, LOGIT_COLOR, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def value(experiment, dataset, kind, metric, N=None):
    sel = curves[(curves.experiment == experiment) & (curves.dataset == dataset)
                 & (curves.kind == kind) & (curves.metric == metric)]
    return sel if N is None else sel[sel.N == N]


# ---- eval-item bootstrap of the snapshot lens (paired with the logit lens) ----
def item_boot(kind, k):
    """{dataset: (J values[B], logit values[B])} for inner pass@k (k=None: mean log10 inner rank)."""
    result = {}
    for dataset in DATASETS:
        sel = ((words.dataset == dataset) & (words.kind == kind)).values
        if not sel.any():
            continue
        _, inv = np.unique(words.item_index.values[sel], return_inverse=True)
        n_items = inv.max() + 1
        per_item = []
        for R in (R_snap, R_logit):
            inner = R[sel, :-1].min(1)
            x = np.log10(inner) if k is None else (inner <= k).astype(float)
            per_item.append(np.bincount(inv, weights=x) / np.bincount(inv))
        draws = rng.integers(0, n_items, size=(args.item_boot, n_items))
        result[dataset] = tuple(p[draws].mean(1) for p in per_item)
    return result


rows = []
for metric, k in (("inner pass@10", 10), ("inner pass@1", 1), ("inner pass@100", 100), ("mean log10 inner rank", None)):
    for kind in ("intermediate", "control", "target"):
        boot_eval = item_boot(kind, k)
        for dataset in DATASETS:
            snap = value("snapshot", dataset, kind, metric)
            if snap.empty:
                continue
            refit = value("order", dataset, kind, metric, N_FULL).value.iloc[0]
            logit = value("logit", dataset, kind, metric).value.iloc[0]
            boot = value("boot", dataset, kind, metric).value.values
            half = value("half", dataset, kind, metric).sort_values("rep").value.values
            half_diff = half[0::2] - half[1::2]
            perm = value("perm", dataset, kind, metric)
            moves = {}
            for n in (32, 64, 96, 112):
                at_n = perm[perm.N == n].value.values
                moves[f"|N{n}-N{N_FULL}|"] = np.mean(np.abs(at_n - refit)) if len(at_n) else np.nan
            J_eval, L_eval = boot_eval[dataset]
            rows.append({
                "metric": metric, "kind": kind, "dataset": dataset,
                "snapshot": snap.value.iloc[0], f"refit N{N_FULL}": refit, "logit lens": logit,
                "fit sd (boot)": boot.std(ddof=1),
                "fit CI lo": np.percentile(boot, 2.5), "fit CI hi": np.percentile(boot, 97.5),
                "fit sd (halves)": half_diff.std(ddof=1) / 2,
                # bias ~ 1/N: (mean over N/2 halves) - full estimates the bias left at N_FULL
                "N/2 bias": half.mean() - refit,
                **moves,
                "eval sd": J_eval.std(ddof=1),
                "eval CI lo": np.percentile(J_eval, 2.5), "eval CI hi": np.percentile(J_eval, 97.5),
                "J-logit": snap.value.iloc[0] - logit,
                "J-logit eval CI lo": np.percentile(J_eval - L_eval, 2.5),
                "J-logit eval CI hi": np.percentile(J_eval - L_eval, 97.5),
            })
table = pd.DataFrame(rows)
table["fit sd / eval sd"] = table["fit sd (boot)"] / table["eval sd"]
table.to_csv(out / "summary.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
main = table[(table.kind == "intermediate") & (table.metric == "inner pass@10")].set_index("dataset")
print(f"=== J-lens, intermediates, inner pass@10 (refit order curve ends at N={N_FULL}) ===")
print(main.drop(columns=["metric", "kind"]).round(3).T.to_string())
for metric in ("mean log10 inner rank", "inner pass@100"):
    sub = table[(table.kind == "intermediate") & (table.metric == metric)].set_index("dataset")
    print(f"\n=== intermediates, {metric} ===")
    print(sub[["snapshot", "fit sd (boot)", "fit sd (halves)", "N/2 bias", f"|N64-N{N_FULL}|", f"|N112-N{N_FULL}|",
               "eval sd", "fit sd / eval sd"]].round(3).T.to_string())
print("\n=== fit sd / eval sd, all kinds and metrics ===")
print(table.pivot_table(index=["metric", "kind"], columns="dataset", values="fit sd / eval sd").round(2).to_string())


# ---- plots: learning curves, one panel per dataset ----
def plot(kind, metric, path, ylabel):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, dataset in zip(axes.flat, DATASETS, strict=False):
        perm = value("perm", dataset, kind, metric)
        band = perm.groupby("N").value.quantile([0.05, 0.95]).unstack()
        ax.fill_between(band.index, band[0.05], band[0.95], color=J_COLOR, alpha=0.18, lw=0,
                        label="J-lens, random prompt orders (5–95%)")
        order = value("order", dataset, kind, metric).sort_values("N")
        ax.plot(order.N, order.value, color=J_COLOR, lw=2, label="J-lens, original fit order")
        boot = value("boot", dataset, kind, metric).value
        ax.errorbar([N_FULL + 4], [order.value.iloc[-1]], yerr=[[order.value.iloc[-1] - boot.quantile(0.025)],
                    [boot.quantile(0.975) - order.value.iloc[-1]]], fmt="o", ms=5, color=INK, capsize=3, lw=1.5,
                    label="fit-corpus bootstrap 95% CI")
        logit = value("logit", dataset, kind, metric).value.iloc[0]
        ax.axhline(logit, color=LOGIT_COLOR, lw=2, ls="--", label="logit lens")
        ax.set_title(dataset, color=INK)
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("fit prompts N", color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


plot("intermediate", "inner pass@10", out / "curve_pass10.png", "inner pass@10, intermediates")
plot("intermediate", "mean log10 inner rank", out / "curve_logrank.png", "mean log10 best inner rank")
plot("control", "inner pass@10", out / "curve_pass10_control.png", "inner pass@10, control")
print("\nwritten", out)
