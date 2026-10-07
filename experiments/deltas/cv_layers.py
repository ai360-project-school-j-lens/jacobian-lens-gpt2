"""Cross-validated layer choice for every lens variant of ``deltas.py``.

pass@k with the minimum over all layers rewards readouts whose hits are scattered
across layers. Here the layer (or a band of ``width`` consecutive layers, scored
by the band's best rank) is chosen on one half of the items -- the band with the
highest intermediate pass@k -- and scored on the other half: intermediates,
controls, and margin = intermediate - control. Halves swap; the estimate averages
both directions over many random splits.

Uncertainty: nested bootstrap -- outer item resampling (copies of an item stay in
one half), inner random splits. All variants share the outer draws, splits and
tie-break jitter, so contrasts are paired.

Writes ``cv_layers.csv`` and ``cv_contrasts.csv`` next to ``deltas_ranks.npz``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convergence"))

import numpy as np
import pandas as pd
from cilib import load_eval_cache

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--widths", type=int, nargs="+", default=[1, 5])
parser.add_argument("--splits", type=int, default=200, help="splits for the point estimate")
parser.add_argument("--boot", type=int, default=1000)
parser.add_argument("--boot-splits", type=int, default=20)
args = parser.parse_args()
out = args.run_dir / "deltas"
rng = np.random.default_rng(3)

words = load_eval_cache(args.run_dir / "eval_cache.pt")["words"]
raw = np.load(out / "deltas_ranks.npz")
ranks = {key.replace("__", "/"): raw[key] for key in raw.files}
variants = list(ranks)
n_inner = next(iter(ranks.values())).shape[1] - 1
DATASETS = list(dict.fromkeys(words.dataset))


def item_hits(R, dataset, kind, width):
    """[n_items, n_bands] share of the item's words with band-best rank <= k; NaN if none."""
    items = np.unique(words.item_index.values[(words.dataset == dataset).values])
    sel = ((words.dataset == dataset) & (words.kind == kind)).values
    inner = R[sel, :n_inner]
    band = np.stack([inner[:, s:s + width].min(1) for s in range(n_inner - width + 1)], axis=1)
    hit = (band <= args.k).astype(float)
    pos = np.searchsorted(items, words.item_index.values[sel])
    total = np.stack([np.bincount(pos, weights=h, minlength=len(items)) for h in hit.T], axis=1)
    count = np.bincount(pos, minlength=len(items))[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def cv(inter, ctrl, counts, masks, jitter):
    """Mean over splits and both directions of (intermediate, control, margin, chosen band).

    counts [n] item multiplicities; masks [S, n] half A; jitter [S, n_bands] tie-break.
    """
    has_i, has_c = ~np.isnan(inter[:, 0]), ~np.isnan(ctrl[:, 0])
    hi, hc = np.nan_to_num(inter), np.nan_to_num(ctrl)
    results = []
    for half in (masks, 1 - masks):
        other = 1 - half
        w_sel = half * counts * has_i
        chosen = np.argmax((w_sel @ hi) / w_sel.sum(1, keepdims=True) + jitter, axis=1)
        w_i, w_c = other * counts * has_i, other * counts * has_c
        rows = np.arange(len(chosen))
        score_i = (w_i @ hi)[rows, chosen] / w_i.sum(1)
        score_c = (w_c @ hc)[rows, chosen] / w_c.sum(1)
        results.append((score_i, score_c, chosen))
    score_i = np.concatenate([r[0] for r in results])
    score_c = np.concatenate([r[1] for r in results])
    chosen = np.concatenate([r[2] for r in results])
    return score_i.mean(), score_c.mean(), (score_i - score_c).mean(), chosen


def half_masks(n, n_splits):
    masks = np.zeros((n_splits, n))
    for s in range(n_splits):
        masks[s, rng.permutation(n)[: n // 2]] = 1
    return masks


rows, boot_rows = [], []
for width in args.widths:
    n_bands = n_inner - width + 1
    for dataset in DATASETS:
        hits = {v: (item_hits(ranks[v], dataset, "intermediate", width),
                    item_hits(ranks[v], dataset, "control", width)) for v in variants}
        n = next(iter(hits.values()))[0].shape[0]
        # point estimate: every item once
        masks = half_masks(n, args.splits)
        jitter = rng.uniform(0, 1e-9, size=(args.splits, n_bands))
        for v in variants:
            inter, ctrl = hits[v]
            score_i, score_c, margin, chosen = cv(inter, ctrl, np.ones(n), masks, jitter)
            in_sample = np.nanmean(inter, axis=0)
            best = int(np.argmax(in_sample))
            rows.append({
                "width": width, "dataset": dataset, "variant": v,
                "cv intermediate": score_i, "cv control": score_c, "cv margin": margin,
                "chosen band start (median)": float(np.median(chosen)),
                "chosen band start (10-90%)": f"{np.percentile(chosen, 10):.0f}-{np.percentile(chosen, 90):.0f}",
                "in-sample best band start": best,
                "in-sample best intermediate": in_sample[best],
                "in-sample best control": np.nanmean(ctrl, axis=0)[best],
            })
        # nested bootstrap, shared draws across variants
        for b in range(args.boot):
            counts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            present = np.flatnonzero(counts)
            masks = np.zeros((args.boot_splits, n))
            for s in range(args.boot_splits):
                masks[s, rng.permutation(present)[: len(present) // 2]] = 1
            jitter = rng.uniform(0, 1e-9, size=(args.boot_splits, n_bands))
            for v in variants:
                score_i, score_c, margin, _ = cv(*hits[v], counts, masks, jitter)
                boot_rows.append({"width": width, "dataset": dataset, "variant": v, "b": b,
                                  "intermediate": score_i, "control": score_c, "margin": margin})
        print(f"width {width} {dataset} done", flush=True)

table = pd.DataFrame(rows)
boot = pd.DataFrame(boot_rows)
ci = boot.groupby(["width", "dataset", "variant"])[["intermediate", "control", "margin"]].quantile([0.025, 0.975]).unstack()
ci.columns = [f"{q} CI {'lo' if p == 0.025 else 'hi'}" for q, p in ci.columns]
table = table.merge(ci.reset_index(), on=["width", "dataset", "variant"])
table.to_csv(out / "cv_layers.csv", index=False)

CONTRASTS = [
    ("J/delta/linear", "J/resid/full"),
    ("logit/delta/linear", "J/resid/full"),
    ("J/delta/linear", "logit/delta/linear"),
    ("J/resid/full", "logit/resid/full"),
    ("logit/delta/linear", "logit/resid/full"),
]
contrast_rows = []
for width in args.widths:
    for dataset in DATASETS:
        sub = boot[(boot.width == width) & (boot.dataset == dataset)].set_index(["variant", "b"])
        point = table[(table.width == width) & (table.dataset == dataset)].set_index("variant")
        for a, b in CONTRASTS:
            for quantity in ("intermediate", "margin"):
                diff = sub.loc[a, quantity].values - sub.loc[b, quantity].values
                contrast_rows.append({
                    "width": width, "dataset": dataset, "contrast": f"{a} − {b}", "quantity": quantity,
                    "estimate": point.loc[a, f"cv {quantity}"] - point.loc[b, f"cv {quantity}"],
                    "CI lo": np.percentile(diff, 2.5), "CI hi": np.percentile(diff, 97.5),
                })
contrasts = pd.DataFrame(contrast_rows)
contrasts.to_csv(out / "cv_contrasts.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
for width in args.widths:
    sub = table[table.width == width]
    for quantity in ("intermediate", "control", "margin"):
        text = sub.assign(text=[f"{x:.3f} [{lo:.3f}, {hi:.3f}]" for x, lo, hi in zip(
            sub[f"cv {quantity}"], sub[f"{quantity} CI lo"], sub[f"{quantity} CI hi"], strict=True)])
        print(f"\n=== width {width}: CV pass@{args.k}, {quantity} (95% CI) ===")
        print(text.pivot(index="variant", columns="dataset", values="text")[DATASETS].to_string())
    print(f"\n=== width {width}: chosen band start, median (10-90%) | in-sample best start, intermediate ===")
    sub = sub.assign(text=[f"{m:.0f} ({r}) | {b} {x:.3f}" for m, r, b, x in zip(
        sub["chosen band start (median)"], sub["chosen band start (10-90%)"],
        sub["in-sample best band start"], sub["in-sample best intermediate"], strict=True)])
    print(sub.pivot(index="variant", columns="dataset", values="text")[DATASETS].to_string())
    c = contrasts[contrasts.width == width]
    c = c.assign(text=[f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" for e, lo, hi in zip(c.estimate, c["CI lo"], c["CI hi"], strict=True)])
    print(f"\n=== width {width}: paired contrasts ===")
    print(c.pivot_table(index=["contrast", "quantity"], columns="dataset", values="text", aggfunc="first")[DATASETS].to_string())
print("\nwritten", out)
