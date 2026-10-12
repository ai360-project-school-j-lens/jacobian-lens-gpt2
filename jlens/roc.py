"""ROC-AUC for separating annotated intermediates from control words by lens rank.

Each supported word is scored by its best (minimum) rank over the selected
layers; a lower rank is a stronger "this word is represented" signal.
Intermediates are positives and controls negatives. Words carry the same
weights as the benchmark's pass@k aggregation: word -> item -> equal item mean,
or for multihop-easy word -> phrasing -> base item -> family -> equal-family
mean. Hence the ROC curve's points at threshold k are exactly
``(control pass@k, intermediate pass@k)``, swept over every k, and the AUC is
the weighted probability that an intermediate outranks a control (ties count
one half). Unsupported (null) ranks are excluded, as in :func:`pass_at_k`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

ARMS = ("J-lens", "logit lens")
EASY_DATASET = "multihop-easy"
_WORD_KEYS = ["dataset", "item", "kind", "word", "role"]


def _best_ranks(ranks: pd.Series, layers: slice | Sequence[int] | None) -> np.ndarray:
    selector = slice(None) if layers is None else layers

    def best(value):
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return np.nan
        selected = np.asarray(value, dtype=float)[selector]
        if not selected.size:
            raise ValueError("Selected layer scope is empty")
        return selected.min()

    return ranks.map(best).to_numpy(dtype=float)


def scored_words(
    words: pd.DataFrame,
    layers: slice | Sequence[int] | None = None,
    easy_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Supported intermediate/control words with best rank, weights and clusters.

    Adds ``best``, ``weight``, ``unit``, ``cluster`` and ``stratum``. Within each
    dataset/lens/kind, weights sum to the number of strata (items count as one
    stratum), so weighted hit rates reproduce the benchmark's pass@k.
    ``weight = within * unit share``: ``within`` averages words in a prompt (and
    phrasings in a multihop-easy base item); the unit share gives each unit
    equal weight within its stratum. Resampling clusters keeps related units.
    """
    if missing := {*_WORD_KEYS, "lens", "ranks"} - set(words):
        raise ValueError(f"Missing rank columns: {sorted(missing)}")
    data = words[words.kind.isin(["intermediate", "control"])].copy()
    data["best"] = _best_ranks(data.ranks, layers)
    data = data[data.best.notna()].drop(columns="ranks")
    easy = data.dataset.eq(EASY_DATASET)
    if easy.any():
        required = ["item", "base_id", "group_id", "family"]
        if easy_metadata is None or not set(required) <= set(easy_metadata):
            raise ValueError("multihop-easy requires base/group/family metadata")
        metadata = easy_metadata[required]
        if metadata.item.duplicated().any() or metadata.isna().any().any():
            raise ValueError("Easy metadata must be non-null and unique by item")
        joined = data.loc[easy, ["item"]].merge(metadata, on="item", how="left")
        if joined[required].isna().any().any():
            raise ValueError("Missing multihop-easy grouping metadata")
        data.loc[easy, "unit"] = joined.base_id.to_numpy()
        data.loc[easy, "cluster"] = joined.group_id.to_numpy()
        data.loc[easy, "stratum"] = joined.family.to_numpy()
    data.loc[~easy, "unit"] = data.loc[~easy, "item"]
    data.loc[~easy, "cluster"] = data.loc[~easy, "item"]
    data.loc[~easy, "stratum"] = "all"
    group = ["dataset", "lens", "kind"]
    words_per_item = data.groupby([*group, "item"]).word.transform("size")
    items_per_unit = data.groupby([*group, "unit"]).item.transform("nunique")
    units_per_stratum = data.groupby([*group, "stratum"]).unit.transform("nunique")
    data["within"] = 1.0 / (words_per_item * items_per_unit)
    data["weight"] = data.within / units_per_stratum
    return data.reset_index(drop=True)


def weighted_roc_auc(
    positive: np.ndarray, positive_weight: np.ndarray,
    negative: np.ndarray, negative_weight: np.ndarray,
) -> float:
    """Weighted P(positive rank < negative rank) + P(tie) / 2; NaN if a class is empty."""
    positive_weight = np.asarray(positive_weight, dtype=float)
    negative_weight = np.asarray(negative_weight, dtype=float)
    total_positive, total_negative = positive_weight.sum(), negative_weight.sum()
    if not len(positive) or not len(negative) or total_positive <= 0 or total_negative <= 0:
        return np.nan
    order = np.argsort(negative, kind="stable")
    sorted_negative = np.asarray(negative, dtype=float)[order]
    cumulative = np.concatenate([[0.0], np.cumsum(negative_weight[order])])
    below = cumulative[np.searchsorted(sorted_negative, positive, side="left")]
    at_or_below = cumulative[np.searchsorted(sorted_negative, positive, side="right")]
    wins = total_negative - at_or_below + 0.5 * (at_or_below - below)
    return float(positive_weight @ wins / (total_positive * total_negative))


def _auc_frame(frame: pd.DataFrame, weight: str = "weight") -> float:
    positive = frame[frame.kind.eq("intermediate")]
    negative = frame[frame.kind.eq("control")]
    return weighted_roc_auc(positive.best.to_numpy(), positive[weight].to_numpy(),
                            negative.best.to_numpy(), negative[weight].to_numpy())


def intermediate_control_roc(
    words: pd.DataFrame,
    layers: slice | Sequence[int] | None = None,
    easy_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """ROC-AUC (intermediate vs control) per dataset and lens, with support counts."""
    data = scored_words(words, layers, easy_metadata)
    rows = []
    for (dataset, lens), group in words.groupby(["dataset", "lens"], sort=True):
        scored = data[data.dataset.eq(dataset) & data.lens.eq(lens)]
        counts = {
            f"n_{kind}_{what}": int(value)
            for kind in ["intermediate", "control"]
            for what, value in [
                ("words", group.kind.eq(kind).sum()),
                ("words_scored", scored.kind.eq(kind).sum()),
                ("items_scored", scored.loc[scored.kind.eq(kind), "item"].nunique()),
            ]
        }
        rows.append({"dataset": dataset, "lens": lens, "roc_auc": _auc_frame(scored), **counts})
    return pd.DataFrame(rows)


def roc_curve(
    words: pd.DataFrame,
    layers: slice | Sequence[int] | None = None,
    easy_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """ROC points per dataset/lens: threshold k, FPR = control pass@k, TPR = intermediate pass@k.

    Rows start at (0, 0) with ``k = 0`` and end at (1, 1) at the largest rank.
    """
    data = scored_words(words, layers, easy_metadata)
    rows = []
    for (dataset, lens), group in data.groupby(["dataset", "lens"], sort=True):
        positive = group[group.kind.eq("intermediate")]
        negative = group[group.kind.eq("control")]
        if positive.empty or negative.empty:
            continue
        thresholds = np.concatenate([[0], np.unique(group.best)])

        def rate(frame, thresholds=thresholds):
            hits = frame.best.to_numpy()[:, None] <= thresholds
            return frame.weight.to_numpy() @ hits / frame.weight.sum()

        rows.append(pd.DataFrame({"dataset": dataset, "lens": lens, "k": thresholds,
                                  "fpr": rate(negative), "tpr": rate(positive)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["dataset", "lens", "k", "fpr", "tpr"])


def roc_bootstrap(
    words: pd.DataFrame,
    layers: slice | Sequence[int] | None = None,
    easy_metadata: pd.DataFrame | None = None,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    **_unused,
) -> pd.DataFrame:
    """Paired cluster bootstrap of intermediate-vs-control ROC-AUC and its J − logit difference.

    Resample clusters with replacement within each stratum (items; multihop-easy
    ``group_id`` within family), with the same draws for both lenses. Controls
    belong to the item whose prompt they were read from, so each draw keeps an
    item's intermediates and controls together. Unit weights are recomputed
    from cluster multiplicities so strata keep equal weight. Draws without both
    classes are skipped and counted. Extra options (e.g. ``batch_size``) are
    accepted for compatibility with :func:`jlens.bootstrap.retrieval_bootstrap`.
    """
    if not isinstance(n_resamples, int) or n_resamples < 2 or not 0 < confidence < 1:
        raise ValueError("Require >=2 resamples and 0<confidence<1")
    data = scored_words(words, layers, easy_metadata)
    alpha = (1 - confidence) / 2
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in sorted(words.dataset.unique()):
        frame = data[data.dataset.eq(dataset) & data.lens.isin(ARMS)]
        estimates = {arm: _auc_frame(frame[frame.lens.eq(arm)]) for arm in ARMS}
        clusters = frame.drop_duplicates("cluster")[["cluster", "stratum"]].reset_index(drop=True)
        cluster_index = pd.Index(clusters.cluster)
        frame = frame.assign(
            cluster_id=cluster_index.get_indexer(frame.cluster),
            key=frame.lens + "\0" + frame.kind + "\0" + frame.stratum.astype(str),
        )
        frame = frame.assign(unit_key=frame.key + "\0" + frame.unit.astype(str))
        key_codes, keys = pd.factorize(frame.key)
        unit_codes, units = pd.factorize(frame.unit_key)
        unit_cluster = np.zeros(len(units), dtype=int)
        unit_cluster[unit_codes] = frame.cluster_id.to_numpy()
        unit_key = np.zeros(len(units), dtype=int)
        unit_key[unit_codes] = key_codes
        strata = [np.asarray(members) for members in clusters.groupby("stratum").indices.values()]
        n_clusters = len(clusters)
        best, within = frame.best.to_numpy(), frame.within.to_numpy()
        masks = [(frame.lens.eq(arm) & frame.kind.eq(kind)).to_numpy()
                 for arm in ARMS for kind in ["intermediate", "control"]]
        draws = []
        if n_clusters >= 2 and not any(np.isnan(list(estimates.values()))):
            for _ in range(n_resamples):
                multiplicity = np.zeros(n_clusters)
                for members in strata:
                    np.add.at(multiplicity, rng.choice(members, size=len(members)), 1)
                unit_mass = multiplicity[unit_cluster]
                denominator = np.bincount(unit_key, weights=unit_mass, minlength=len(keys))
                share = np.divide(unit_mass, denominator[unit_key],
                                  out=np.zeros_like(unit_mass), where=denominator[unit_key] > 0)
                weight = within * share[unit_codes]
                draws.append([
                    weighted_roc_auc(best[positive], weight[positive], best[negative], weight[negative])
                    for positive, negative in zip(masks[::2], masks[1::2], strict=True)
                ])
        draws = np.asarray(draws, dtype=float).reshape(-1, len(ARMS))
        valid = draws[np.isfinite(draws).all(axis=1)]
        statistics = [(arm, "", estimates[arm], valid[:, i]) for i, arm in enumerate(ARMS)]
        statistics.append((ARMS[0], ARMS[1], estimates[ARMS[0]] - estimates[ARMS[1]],
                           valid[:, 0] - valid[:, 1]))
        for arm, reference, estimate, distribution in statistics:
            if np.isnan(estimate):
                status = "no_supported_units"
            elif n_clusters < 2 or len(valid) < 2:
                status = "too_few_clusters"
            elif np.ptp(distribution) <= 1e-12:
                status = "degenerate"
            else:
                status = "ok"
            low, high = (np.quantile(distribution, [alpha, 1 - alpha])
                         if status == "ok" else (np.nan, np.nan))
            rows.append({
                "dataset": dataset, "statistic": "difference" if reference else "arm",
                "arm": arm, "reference": reference, "estimate": estimate,
                "ci_low": low, "ci_high": high, "status": status,
                "n_clusters": n_clusters, "n_resamples": n_resamples,
                "n_valid_resamples": len(valid), "confidence": confidence,
            })
    return pd.DataFrame(rows)
