"""Digitize the bootstrap pass@k figures of the latent-bridge benchmark notebooks.

The interval CSVs of the GPT-2 XL and Gemma 4 31B runs exist only on the
machines that ran them, so their curves are read back from the PNG figures
saved in the notebooks. The figures come from one known matplotlib recipe
(`latent_bridge_v2_lens_benchmark_bootstrap.ipynb`, pass@k cell):

* panels in sorted dataset order, shared log x axis over k = 1..100 with
  default 5% margins, fixed y limits (-0.02, 1.02);
* J-lens #2a78d6 and logit lens #eb6834; intermediate solid, control dashed;
* each 95% band is `fill_between(..., alpha=0.12)`, drawn after its line, in the
  order J control, J intermediate, logit control, logit intermediate.

Lines are tracked column by column (two tracks per color); within each segment
between sampled k the solid track is the one with fewer gaps. Bands are read by
matching each pixel to the 9 possible alpha-blend colors. Values are accurate to
roughly one pixel (~0.005 in pass@k); `validate` compares them with exact data.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KS = np.array([1, 2, 5, 10, 20, 50, 100])
YLIM = (-0.02, 1.02)
LOG_XLIM = (np.log10(1) - 0.1, np.log10(100) + 0.1)  # matplotlib's 5% margins
COLORS = {"J-lens": np.array([42, 120, 214.0]), "logit lens": np.array([235, 104, 52.0])}
ALPHA = 0.12
DATASETS = {
    "gpt2-xl": ["association", "latent bridge: all but slot", "multihop", "multihop-easy",
                "multihop-easy-fewshot", "multilingual", "order-ops", "poetry", "typo"],
    "gemma-4-31b": ["association", "latent bridge: all but slot", "multihop", "multihop-easy",
                    "multilingual", "order-ops", "poetry", "typo"],
}
DATASETS["gemma-4-e4b"] = DATASETS["gemma-4-31b"]


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=float)


def find_panels(image: np.ndarray, min_length: int = 250) -> list[tuple[float, float, float, float]]:
    """Return (left, right, top, bottom) spine centers of every axes, row-major."""
    dark = image.max(axis=2) < 110
    spans = []  # (row, start, end) of long horizontal dark runs
    for row in range(dark.shape[0]):
        cols = np.flatnonzero(dark[row])
        if len(cols) < min_length:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1)
        for seg in np.split(cols, breaks + 1):
            if len(seg) >= min_length:
                spans.append((row, seg[0], seg[-1]))
    panels = []
    for (start, end), group in itertools.groupby(
        sorted(spans, key=lambda s: (round(s[1] / 4), round(s[2] / 4), s[0])),
        key=lambda s: (round(s[1] / 4), round(s[2] / 4)),
    ):
        rows = [s[0] for s in group]
        # Merge adjacent anti-aliased rows into single spine positions.
        lines, current = [], [rows[0]]
        for r in rows[1:]:
            if r - current[-1] <= 2:
                current.append(r)
            else:
                lines.append(np.mean(current))
                current = [r]
        lines.append(np.mean(current))
        for top, bottom in zip(lines[0::2], lines[1::2], strict=True):
            panels.append((start * 4.0, end * 4.0, top, bottom))
    # Refine left/right from the dark vertical spines near the rough edges.
    refined = []
    for left, right, top, bottom in panels:
        rows = slice(int(top) + 3, int(bottom) - 2)
        columns = dark[rows].mean(axis=0)
        def spine(near):
            window = np.arange(max(0, int(near) - 8), min(dark.shape[1], int(near) + 9))
            hits = window[columns[window] > 0.8]
            return hits.mean() if len(hits) else near
        refined.append((spine(left), spine(right), top, bottom))
    return sorted(refined, key=lambda p: (round(p[2] / 20), p[0]))


def blend_palette() -> dict[tuple[int, int], np.ndarray]:
    """Background colors for nJ, nL overlapping 12% bands (J bands drawn first)."""
    palette = {}
    for n_j, n_l in itertools.product(range(3), range(3)):
        color = np.full(3, 255.0)
        for _ in range(n_j):
            color = (1 - ALPHA) * color + ALPHA * COLORS["J-lens"]
        for _ in range(n_l):
            color = (1 - ALPHA) * color + ALPHA * COLORS["logit lens"]
        palette[n_j, n_l] = color
    return palette


PALETTE = blend_palette()


def line_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    distances = {arm: np.linalg.norm(image - c, axis=2) for arm, c in COLORS.items()}
    return {arm: (d < 75) & (d < distances[other])
            for (arm, d), other in zip(distances.items(), ["logit lens", "J-lens"], strict=True)}


def column_runs(mask_column: np.ndarray, offset: int) -> list[float]:
    rows = np.flatnonzero(mask_column)
    if not len(rows):
        return []
    # Gaps up to 4 px close runs split by a crossing line of the other color.
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 4) + 1)
    return [offset + g.mean() for g in groups]


def track(mask: np.ndarray, columns: range, top: int, bottom: int) -> list[dict[int, float]]:
    """Follow the two same-colored curves (solid and dashed) from left to right.

    While only one curve has been seen, both tracks share it; the first run far
    from it starts the second track. Two runs in a column are always the two
    curves. A single run joins the nearer track, with a tolerance that grows
    over gaps (dashes, or a curve hidden under the other color's line).
    """
    tracks: list[dict[int, float]] = [{}, {}]
    shared = True  # the second curve has not been seen separately yet

    def predict(t: dict[int, float], x: int) -> tuple[float, int]:
        xs = sorted(t)[-6:]
        last = xs[-1]
        if len(xs) >= 2 and xs[-1] - xs[0] > 0:
            slope = (t[xs[-1]] - t[xs[0]]) / (xs[-1] - xs[0])
            return t[last] + slope * (x - last), x - last
        return t[last], x - last

    for x in columns:
        runs = column_runs(mask[top:bottom, x], top)
        if not runs:
            continue
        if not tracks[0]:
            tracks[0][x] = runs[0]
            tracks[1][x] = runs[-1]
            shared = len(runs) < 2
            continue
        (p0, gap0), (p1, gap1) = predict(tracks[0], x), predict(tracks[1], x)
        if shared:
            if len(runs) >= 2:
                near = int(np.argmin([abs(r - p0) for r in runs]))
                far = int(np.argmax([abs(r - runs[near]) for r in runs]))
                tracks[0][x], tracks[1][x] = runs[near], runs[far]
                shared = False
            elif abs(runs[0] - p0) <= 12 + 0.5 * gap0:
                tracks[0][x] = tracks[1][x] = runs[0]
            else:
                tracks[1][x] = runs[0]
                shared = False
            continue
        if len(runs) >= 2:
            pairs = itertools.permutations(range(len(runs)), 2)
            a, b = min(pairs, key=lambda ab: abs(runs[ab[0]] - p0) + abs(runs[ab[1]] - p1))
            tracks[0][x], tracks[1][x] = runs[a], runs[b]
        elif abs(p0 - p1) < 3:
            tracks[0][x] = tracks[1][x] = runs[0]
        else:
            d0, d1 = abs(runs[0] - p0), abs(runs[0] - p1)
            best, dist, gap = (0, d0, gap0) if d0 <= d1 else (1, d1, gap1)
            if dist <= 12 + 0.5 * gap:
                tracks[best][x] = runs[0]
    return tracks


def segment_line(t: dict[int, float], lo: float, hi: float) -> np.ndarray | None:
    """Robust straight-line fit (y = a x + b) to a track's points inside a segment."""
    xs = np.array([x for x in t if lo + 1 <= x <= hi - 1], dtype=float)
    if len(xs) < 3:
        return None
    ys = np.array([t[int(x)] for x in xs])
    fit = np.polyfit(xs, ys, 1)
    for _ in range(2):
        keep = np.abs(np.polyval(fit, xs) - ys) <= 2.5
        if keep.sum() < 3:
            break
        fit = np.polyfit(xs[keep], ys[keep], 1)
    return fit


def classify_bands(image: np.ndarray, x: int, top: int, bottom: int) -> np.ndarray:
    """Per-row (nJ, nL) band counts in one column; line pixels inherit neighbors."""
    column = image[top:bottom, x]
    keys = list(PALETTE)
    colors = np.stack([PALETTE[k] for k in keys])
    distances = np.linalg.norm(column[:, None, :] - colors[None], axis=2)
    best = distances.argmin(axis=1)
    counts = np.array([keys[i] for i in best], dtype=float)
    unknown = distances.min(axis=1) > 6  # lines, anti-aliasing, text
    counts[unknown] = np.nan
    frame = pd.DataFrame(counts).ffill().bfill()
    # Fill each unknown run from the nearer side to avoid smearing across lines.
    return frame.to_numpy()


def band_interval(counts: np.ndarray, lens_index: int, y_line: float, other_y: float | None,
                  top: int) -> tuple[float, float] | None:
    """Pixel extent [upper, lower] of the band containing a line at y_line."""
    covered = counts[:, lens_index] >= 1
    row = int(round(y_line)) - top
    rows = np.flatnonzero(covered)
    if not len(rows):
        return None
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    group = next((g for g in groups if g[0] - 2 <= row <= g[-1] + 2), None)
    if group is None:
        return None
    return group[0] - 0.5 + top, group[-1] + 0.5 + top


def clean_profile(profile: np.ndarray, min_run: int = 3) -> np.ndarray:
    """Merge coverage runs shorter than min_run px into their longer neighbor.

    Anti-aliased band and line edges produce 1-2 px rows that happen to match
    another blend color; real bands are always taller than that.
    """
    p = np.nan_to_num(profile, nan=0).astype(int)
    while True:
        cuts = np.flatnonzero(np.diff(p)) + 1
        starts, ends = np.concatenate([[0], cuts]), np.concatenate([cuts, [len(p)]])
        short = [i for i in range(len(starts)) if ends[i] - starts[i] < min_run]
        if not short or len(starts) == 1:
            return p
        i = min(short, key=lambda j: ends[j] - starts[j])
        left = ends[i - 1] - starts[i - 1] if i > 0 else -1
        right = ends[i + 1] - starts[i + 1] if i + 1 < len(starts) else -1
        p[starts[i]:ends[i]] = p[starts[i] - 1] if left >= right else p[ends[i]]


def band_edges(profile: np.ndarray, top: int) -> tuple[list[float], list[float]] | None:
    """Band tops and bottoms (pixel boundaries) from a 0/1/2 coverage profile."""
    p = clean_profile(profile)
    padded = np.concatenate([[0], p, [0]])
    tops, bottoms = [], []
    for i in np.flatnonzero(np.diff(padded)):
        delta = padded[i + 1] - padded[i]
        boundary = i - 0.5 + top
        (tops if delta > 0 else bottoms).extend([boundary] * abs(delta))
    if len(tops) != len(bottoms) or len(tops) > 2:
        return None
    return tops, bottoms


def assign_bands(tops: list[float], bottoms: list[float],
                 lines: dict[str, float]) -> dict[str, tuple[float, float] | None]:
    """Pair band edges into bands and give each line the band that contains it.

    Two bands can be side by side (T1,B1)+(T2,B2) or nested (T1,B2)+(T2,B1);
    among pairings where every line lies inside its band, prefer the one that
    centers lines in their bands (bootstrap percentile bands are near-symmetric).
    """
    kinds = [k for k, y in lines.items() if not np.isnan(y)]
    result: dict[str, tuple[float, float] | None] = {k: None for k in lines}
    if not tops or not kinds:
        return result
    bands = [(tops[0], bottoms[0])]
    options = [[(tops[0], bottoms[0])]]
    if len(tops) == 2:
        options = [[(tops[0], bottoms[0]), (tops[1], bottoms[1])]]
        if tops[1] < bottoms[0]:
            options.append([(tops[0], bottoms[1]), (tops[1], bottoms[0])])
    best, best_score = None, np.inf
    for bands in options:
        for chosen in itertools.permutations(range(len(bands)), min(len(bands), len(kinds))):
            mapping = dict(zip(kinds, chosen, strict=False))
            score, valid = 0.0, True
            for kind, b in mapping.items():
                t, bt = bands[b]
                y = lines[kind]
                valid &= t - 2 <= y <= bt + 2
                score += abs(y - (t + bt) / 2) / max(bt - t, 1)
            # One region and two lines: identical bands, touching bands, or one
            # line without a band (degenerate interval).
            if len(bands) == 1 and len(kinds) == 2:
                inside = [k for k in kinds if bands[0][0] - 2 <= lines[k] <= bands[0][1] + 2]
                if len(inside) == 2:
                    return split_region(bands[0], lines)
                mapping = {k: 0 for k in inside}
                valid = bool(inside)
            if valid and score < best_score:
                best, best_score = {k: bands[b] for k, b in mapping.items()}, score
    if best is None:
        return result
    result.update(best)
    return result


def split_region(region: tuple[float, float],
                 lines: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Share one covered region between two lines inside it.

    Coincident lines get the whole region. Otherwise the two bands touch without
    a visible overlap, so the boundary is placed midway between the lines.
    """
    (k1, y1), (k2, y2) = sorted(lines.items(), key=lambda kv: kv[1])
    if y2 - y1 <= 2:
        return {k1: region, k2: region}
    middle = (y1 + y2) / 2
    return {k1: (region[0], middle), k2: (middle, region[1])}


def column_bands(image: np.ndarray, x: int, t0: int, t1: int, lens_index: int,
                 lines: dict[str, float]) -> dict[str, tuple[float, float] | None]:
    counts = classify_bands(image, x, t0, t1)
    edges = band_edges(counts[:, lens_index], t0)
    if edges is not None:
        return assign_bands(*edges, lines)
    # Fallback: covered regions containing each line, split where shared.
    covered = clean_profile(counts[:, lens_index]) >= 1
    rows = np.flatnonzero(covered)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if len(rows) else []
    result: dict[str, tuple[float, float] | None] = {k: None for k in lines}
    regions = {}
    for kind, y in lines.items():
        if np.isnan(y):
            continue
        g = next((g for g in groups if g[0] - 2 <= y - t0 <= g[-1] + 2), None)
        if g is not None:
            regions.setdefault((g[0] - 0.5 + t0, g[-1] + 0.5 + t0), []).append(kind)
    for region, kinds in regions.items():
        if len(kinds) == 2:
            result.update(split_region(region, {k: lines[k] for k in kinds}))
        else:
            result[kinds[0]] = region
    return result


def digitize(model: str, path: Path) -> pd.DataFrame:
    image = load(path)
    panels = find_panels(image)
    names = DATASETS[model]
    if len(panels) != len(names):
        raise ValueError(f"{model}: found {len(panels)} panels, expected {len(names)}")
    masks = line_masks(image)
    rows = []
    for dataset, (left, right, top, bottom) in zip(names, panels, strict=True):
        def x_of(k):
            return left + (np.log10(k) - LOG_XLIM[0]) / (LOG_XLIM[1] - LOG_XLIM[0]) * (right - left)

        def value(y):
            return YLIM[0] + (bottom - y) / (bottom - top) * (YLIM[1] - YLIM[0])

        x_k = [x_of(k) for k in KS]
        t0, t1 = int(top) + 2, int(bottom) - 1
        cols = range(int(x_k[0]) - 2, int(x_k[-1]) + 3)
        vertex = {}  # (arm, kind) -> array of y pixel per k (NaN where unseen)
        for arm in ["J-lens", "logit lens"]:
            tracks = track(masks[arm], cols, t0, t1)
            ends = {kind: [[] for _ in KS] for kind in ("intermediate", "control")}
            for seg, (a, b) in enumerate(zip(x_k[:-1], x_k[1:], strict=True)):
                fits = [segment_line(t, a, b) for t in tracks]
                span = range(int(a) + 2, int(b) - 1)
                apart = [x for x in span if x in tracks[0] and x in tracks[1]
                         and abs(tracks[0][x] - tracks[1][x]) > 5]
                if len(apart) >= 4:
                    # The solid (intermediate) line has fewer gaps where the two differ.
                    window = range(min(apart), max(apart) + 1)
                    fill = [sum(x in t for x in window) for t in tracks]
                    solid = 0 if fill[0] >= fill[1] else 1
                    assignments = (("intermediate", solid), ("control", 1 - solid))
                else:
                    # Only one curve visible: solid if it covers nearly every column,
                    # otherwise dashed. The unseen one is hidden or coincident and is
                    # filled below from the other color (or this color's other kind).
                    coverage = sum(x in tracks[0] for x in span) / max(1, len(span))
                    assignments = ((("intermediate" if coverage >= 0.85 else "control"), 0),)
                for kind, index in assignments:
                    if fits[index] is not None:
                        ends[kind][seg].append(np.polyval(fits[index], a))
                        ends[kind][seg + 1].append(np.polyval(fits[index], b))
            for kind, values in ends.items():
                vertex[arm, kind] = np.array([np.mean(v) if v else np.nan for v in values])
        # A curve unseen near a vertex is hidden under the other color's line of
        # the same kind (drawn later, same y within a line width).
        # If that is unseen too, the curve coincides with this color's other kind.
        occluded = {}
        seen = {key: ~np.isnan(v) for key, v in vertex.items()}
        for arm, other in (("J-lens", "logit lens"), ("logit lens", "J-lens")):
            for kind, other_kind in (("intermediate", "control"), ("control", "intermediate")):
                own = vertex[arm, kind]
                fill = np.where(seen[other, kind], vertex[other, kind], vertex[arm, other_kind])
                occluded[arm, kind] = ~seen[arm, kind]
                vertex[arm, kind] = np.where(seen[arm, kind], own, fill)
        for lens_index, arm in enumerate(["J-lens", "logit lens"]):
            for i, (k, xv) in enumerate(zip(KS, x_k, strict=True)):
                lines = {kind: vertex[arm, kind][i] for kind in ("intermediate", "control")}
                # Band polygons have vertices at xv; sample 2 px inside each side and
                # average (interior) or extrapolate linearly (first and last k).
                if i == 0:
                    samples, weights = [round(xv) + 2, round(xv) + 4], [2.0, -1.0]
                elif i == len(KS) - 1:
                    samples, weights = [round(xv) - 2, round(xv) - 4], [2.0, -1.0]
                else:
                    samples, weights = [round(xv) - 2, round(xv) + 2], [0.5, 0.5]
                found = [column_bands(image, int(x), t0, t1, lens_index, lines) for x in samples]
                for kind in ("intermediate", "control"):
                    y = lines[kind]
                    edges = [f[kind] for f in found]
                    if np.isnan(y) or all(e is None for e in edges):
                        interval = None
                    elif any(e is None for e in edges):
                        interval = next(e for e in edges if e is not None)
                    else:
                        interval = tuple(sum(w * e[j] for w, e in zip(weights, edges, strict=True))
                                         for j in (0, 1))
                    rows.append(dict(
                        model=model, dataset=dataset, arm=arm, kind=kind, k=int(k),
                        estimate=np.nan if np.isnan(y) else value(y),
                        ci_low=np.nan if interval is None else value(interval[1]),
                        ci_high=np.nan if interval is None else value(interval[0]),
                        occluded=bool(occluded[arm, kind][i]),
                    ))
    frame = pd.DataFrame(rows)
    frame[["estimate", "ci_low", "ci_high"]] = frame[["estimate", "ci_low", "ci_high"]].clip(0, 1)
    return frame


def validate(digitized: pd.DataFrame, exact: pd.DataFrame) -> pd.DataFrame:
    """Absolute errors of digitized values against exact rows (matching k)."""
    keys = ["model", "dataset", "arm", "kind", "k"]
    merged = digitized.merge(exact[keys + ["estimate", "ci_low", "ci_high"]], on=keys,
                             suffixes=("", "_exact"))
    for column in ("estimate", "ci_low", "ci_high"):
        merged[f"err_{column}"] = (merged[column] - merged[f"{column}_exact"]).abs()
    return merged


def main() -> None:
    frames = []
    for model in DATASETS:
        path = ROOT / f"poster/source_figures/{model}_bootstrap_pass_at_k.png"
        frames.append(digitize(model, path))
    digitized = pd.concat(frames, ignore_index=True)
    out = ROOT / "poster/data/digitized_pass_at_k.csv"
    digitized.to_csv(out, index=False)
    print("saved", out.relative_to(ROOT), len(digitized), "rows")

    exact = pd.read_csv(ROOT / "poster/data/exact_pass_at_10.csv")
    exact = exact[exact.statistic.eq("arm")]
    check = validate(digitized, exact)
    errors = ["err_estimate", "err_ci_low", "err_ci_high"]
    print("\nAbsolute error at k = 10 vs exact notebook tables (pass@k units):")
    print(check.groupby("model")[errors].agg(["mean", "max"]).round(4).to_string())
    worst = check.assign(worst=check[errors].max(axis=1)).nlargest(8, "worst")
    print("\nLargest deviations:")
    print(worst[["model", "dataset", "arm", "kind", "estimate", "estimate_exact", "ci_low",
                 "ci_low_exact", "ci_high", "ci_high_exact"]].round(3).to_string(index=False))

    full = ROOT / "poster/data/gemma-4-e4b_bootstrap_intervals_exact.csv"
    if full.exists():
        e4b = pd.read_csv(full)
        e4b = e4b[e4b.statistic.eq("arm") & e4b.metric.eq("pass_at_k") & e4b.split.eq("all")]
        e4b = e4b.assign(model="gemma-4-e4b")
        check = validate(digitized, e4b)
        print("\nGemma E4B, all k vs exact interval CSV:")
        print(check[errors].agg(["mean", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
