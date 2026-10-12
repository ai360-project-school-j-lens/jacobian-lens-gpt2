"""Digitize the token-swap layer-band bar charts and extract exact primary-band values.

The bootstrap interval CSVs of the GPT-2 XL and Gemma 4 31B token-swap runs
exist only on the machines that ran them, so per-band values are read back from
the saved chart (`model_agnostic_token_swap_bootstrap.ipynb`, layer-band cell):

* one panel per dataset, pandas bars for J / logit / random (C0, C1, C2) at
  x = band index + (-1/6, 0, 1/6), x limits (-0.5, 2.5), y limits (0, 1);
* black percentile error bars at the bar center; zero bars are invisible and
  have degenerate (null) intervals.

The primary (middle) band is taken from the exact interval table saved in the
notebook (cell 18); it also serves to check the digitized values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from digitize_bootstrap_figures import find_panels, load
from extract_exact_k10 import first_html_table

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "gpt2-xl": "notebooks/gpt2-xl/token_swap/gpt2xl_token_swap_bootstrap.ipynb",
    "gemma-4-31b": "notebooks/gemma-4-31b/token_swap/gemma31b_model_agnostic_token_swap_bootstrap.ipynb",
    "gemma-4-e4b": "notebooks/gemma-4-e4b/token_swap/model_agnostic_token_swap_bootstrap.ipynb",
}
DATASETS = ["flexible-generalization", "latent-bridge", "probe-swap"]
BANDS = ["early", "middle", "late"]  # 20-40%, 40-60%, 60-80% of depth (intermediate_bands)
ARMS = {"J": np.array([31, 119, 180.0]), "logit": np.array([255, 127, 14.0]),
        "random": np.array([44, 160, 44.0])}


def digitize(model: str) -> pd.DataFrame:
    image = load(ROOT / f"poster/source_figures/{model}_token_swap_layer_bands.png")
    panels = find_panels(image, min_length=300)
    if len(panels) != len(DATASETS):
        raise ValueError(f"{model}: found {len(panels)} panels")
    dark = image.max(axis=2) < 90
    rows = []
    for dataset, (left, right, top, bottom) in zip(DATASETS, panels, strict=True):
        def x_px(x):
            return left + (x + 0.5) / 3 * (right - left)

        def value(y):
            return (bottom - y) / (bottom - top)

        inner = slice(int(top) + 2, int(bottom) - 1)
        for b, band in enumerate(BANDS):
            for a, (arm, color) in enumerate(ARMS.items()):
                center = x_px(b + (a - 1) / 6)
                # Bar color away from the centered error bar and its caps.
                side = [int(round(center + d)) for d in (-7, -6, 6, 7)]
                hits = [np.flatnonzero(np.linalg.norm(image[inner, x] - color, axis=1) < 40)
                        for x in side]
                tops = [h[0] for h in hits if len(h) >= 2]
                if not tops:
                    rows.append(dict(model=model, dataset=dataset, band=band, arm=arm,
                                     estimate=0.0, ci_low=np.nan, ci_high=np.nan))
                    continue
                bar_top = np.median(tops) + int(top) + 2 - 0.5
                column = dark[inner, int(round(center)) - 1:int(round(center)) + 2].any(axis=1)
                runs = np.flatnonzero(column)
                ci = (np.nan, np.nan)
                if len(runs):
                    groups = np.split(runs, np.flatnonzero(np.diff(runs) > 1) + 1)
                    # The error bar is the dark run that spans the bar top.
                    row_top = bar_top - (int(top) + 2)
                    g = next((g for g in groups if g[0] - 3 <= row_top <= g[-1] + 3), None)
                    if g is not None and len(g) >= 3:
                        hi = g[0] + int(top) + 2
                        lo = g[-1] + int(top) + 2
                        ci = (max(0.0, value(lo + 0.5)), min(1.0, value(hi - 0.5)))
                rows.append(dict(model=model, dataset=dataset, band=band, arm=arm,
                                 estimate=float(np.clip(value(bar_top), 0, 1)),
                                 ci_low=ci[0], ci_high=ci[1]))
    return pd.DataFrame(rows)


def exact_primary(model: str) -> pd.DataFrame:
    cells = json.loads((ROOT / NOTEBOOKS[model]).read_text())["cells"]
    table = first_html_table(cells[18])
    for column in ("estimate", "ci_low", "ci_high", "n_units", "n_clusters", "n_trials"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table.assign(model=model, band="middle")


def main() -> None:
    digitized = pd.concat([digitize(m) for m in NOTEBOOKS], ignore_index=True)
    exact = pd.concat([exact_primary(m) for m in NOTEBOOKS], ignore_index=True)
    digitized.to_csv(ROOT / "poster/data/token_swap_digitized.csv", index=False)
    exact.to_csv(ROOT / "poster/data/token_swap_exact_primary.csv", index=False)
    arms = exact[exact.statistic.eq("arm")]
    check = digitized.merge(arms[["model", "dataset", "band", "arm", "estimate", "ci_low", "ci_high"]],
                            on=["model", "dataset", "band", "arm"], suffixes=("", "_exact"))
    for column in ("estimate", "ci_low", "ci_high"):
        check[f"err_{column}"] = (check[column] - check[f"{column}_exact"]).abs()
    errors = ["err_estimate", "err_ci_low", "err_ci_high"]
    print("Primary band, digitized vs exact (absolute error):")
    print(check.groupby("model")[errors].agg(["mean", "max"]).round(4).to_string())
    print(check[["model", "dataset", "arm", "estimate", "estimate_exact", "ci_low", "ci_low_exact",
                 "ci_high", "ci_high_exact"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
