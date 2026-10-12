"""Paired uncertainty estimates from cached lens and intervention measurements.

The resampling unit is a prompt, or a group of related base items. Methods are
always resampled together. These are pointwise percentile intervals conditional
on a fixed model, lens, dataset, scoring rule, and intervention setting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def paired_cluster_bootstrap(
    scores: pd.DataFrame,
    *,
    arms: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Bootstrap equal-stratum means, preserving paired arms and whole clusters.

    Input columns are ``unit, cluster, stratum, arm, value``, with one row per
    unit/arm. Units have equal weight within a stratum; strata have equal weight.
    Clusters must be nested within strata. Within each stratum, sample its
    original number of clusters uniformly with replacement, retaining every
    unit of a sampled cluster and its multiplicity. A cluster can contain
    multiple units (e.g. multiple multihop-easy base items for one concept).

    Missing values must agree across arms; entirely unsupported units are
    excluded and counted. Unpaired rows, duplicate units, and nonfinite values
    are errors. Insufficient clusters and degenerate bootstrap distributions
    have null interval endpoints and an explicit status, including all-zero
    controls. They must not be interpreted as population certainty.
    """
    arms = list(arms)
    comparisons = list(comparisons)
    if len(arms) < 2 or len(set(arms)) != len(arms):
        raise ValueError("Require at least two distinct arms")
    if any(a not in arms or b not in arms or a == b for a, b in comparisons):
        raise ValueError("Comparisons must name distinct requested arms")
    if (
        not isinstance(n_resamples, int) or n_resamples < 2
        or not isinstance(batch_size, int) or batch_size < 1
        or not 0 < confidence < 1
    ):
        raise ValueError("Require >=2 resamples, positive batch size, and 0<confidence<1")
    required = ["unit", "cluster", "stratum", "arm", "value"]
    if missing := set(required) - set(scores):
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    data = scores.loc[scores.arm.isin(arms), required].copy()
    if data[required[:-1]].isna().any().any():
        raise ValueError("Unit, cluster, stratum, and arm cannot be null")
    if data.duplicated(["unit", "arm"]).any():
        raise ValueError("Duplicate unit/arm scores; aggregate repeated trials first")
    if not data.groupby("unit").arm.nunique().eq(len(arms)).all():
        raise ValueError("Every unit must have all paired arms")
    if (data.groupby("unit")[["cluster", "stratum"]].nunique() > 1).any().any():
        raise ValueError("Cluster/stratum metadata must agree across paired arms")
    if (data.groupby("cluster").stratum.nunique() > 1).any():
        raise ValueError("Clusters must be nested within strata")
    data["value"] = pd.to_numeric(data.value, errors="raise")
    if np.isinf(data.value.to_numpy(dtype=float)).any():
        raise ValueError("Scores must be finite or null")

    wide = data.pivot(index="unit", columns="arm", values="value").reindex(columns=arms)
    support = wide.notna()
    if not support.eq(support.iloc[:, 0], axis=0).all().all():
        raise ValueError("Paired arms have different support; cannot silently drop pairs")
    metadata = data.drop_duplicates("unit").set_index("unit")[["cluster", "stratum"]]
    total_strata = metadata.stratum.nunique()
    n_total = len(wide)
    wide = wide.loc[support.all(axis=1)]
    metadata = metadata.loc[wide.index]
    plans = []
    for _, group in metadata.groupby("stratum", sort=True):
        values = wide.loc[group.index].to_numpy(dtype=float)
        cluster_ids, names = pd.factorize(group.cluster, sort=True)
        sums = np.zeros((len(names), len(arms)))
        np.add.at(sums, cluster_ids, values)
        counts = np.bincount(cluster_ids).astype(float)
        plans.append((sums, counts))

    n_columns = len(arms) + len(comparisons)
    points = np.full(n_columns, np.nan)
    if plans:
        points[:len(arms)] = np.mean(
            [sums.sum(axis=0) / counts.sum() for sums, counts in plans], axis=0,
        )
        for j, (a, b) in enumerate(comparisons, len(arms)):
            points[j] = points[arms.index(a)] - points[arms.index(b)]
    distributions = np.empty((n_resamples, n_columns))
    enough = bool(plans) and all(len(counts) >= 2 for _, counts in plans)
    if enough:
        # Separate stratum RNGs keep results invariant to the memory batch size.
        generators = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(plans))]
        for start in range(0, n_resamples, batch_size):
            stop = min(n_resamples, start + batch_size)
            means = np.zeros((stop - start, len(arms)))
            for (sums, counts), rng in zip(plans, generators, strict=True):
                size = len(counts)
                weights = rng.multinomial(size, np.full(size, 1 / size), size=stop - start)
                means += (weights @ sums) / (weights @ counts)[:, None] / len(plans)
            distributions[start:stop, :len(arms)] = means
            for j, (a, b) in enumerate(comparisons, len(arms)):
                distributions[start:stop, j] = means[:, arms.index(a)] - means[:, arms.index(b)]

    rows = []
    labels = [("arm", a, "") for a in arms] + [("difference", a, b) for a, b in comparisons]
    for j, (statistic, arm, reference) in enumerate(labels):
        low = high = error = np.nan
        if not plans:
            status = "no_supported_units"
        elif not enough:
            status = "too_few_clusters"
        elif np.ptp(distributions[:, j]) <= 1e-12:
            status = "degenerate"
        else:
            status = "ok"
            tail = (1 - confidence) / 2
            low, high = np.quantile(distributions[:, j], [tail, 1 - tail])
            error = distributions[:, j].std(ddof=1)
        rows.append(dict(
            statistic=statistic, arm=arm, reference=reference,
            estimate=points[j], ci_low=low, ci_high=high, standard_error=error,
            status=status, n_units_total=n_total, n_units=len(wide),
            n_units_excluded=n_total - len(wide), n_clusters=metadata.cluster.nunique(),
            n_strata_total=total_strata, n_strata=len(plans),
            n_resamples=n_resamples, confidence=confidence, seed=seed,
            interval_method="paired stratified cluster percentile",
        ))
    return pd.DataFrame(rows)


def swap_bootstrap(results: pd.DataFrame, **bootstrap_options) -> pd.DataFrame:
    """Summarize cached token swaps with prompt-clustered rates and paired gains.

    Each setting is assessed independently, never selected by observed success.
    Capability filtering is on the shared clean baseline. Average replacement
    targets within prompt before bootstrapping; do not treat them as independent.
    """
    keys = ["dataset", "band", "alpha"]
    arms = ["J", "logit", "random"]
    metrics = ["success", "changed"]
    required = {*keys, "trial_id", "prompt", "mode", "base_correct", *metrics}
    if missing := required - set(results):
        raise ValueError(f"Missing swap columns: {sorted(missing)}")
    data = results[results["mode"].isin(arms)].copy()
    if data[list(required)].isna().any().any():
        raise ValueError("Swap identities and outcomes cannot be null")
    for column in ["base_correct", *metrics]:
        if not data[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain boolean outcomes")
    if data.duplicated([*keys, "trial_id", "mode"]).any():
        raise ValueError("Duplicate swap trial/setting/arm")
    if not data.groupby([*keys, "trial_id"])["mode"].nunique().eq(3).all():
        raise ValueError("Every swap trial must have all paired arms")
    if (data.groupby(["dataset", "trial_id"])[["prompt", "base_correct"]].nunique() > 1).any().any():
        raise ValueError("Prompt/baseline must agree across arms and settings")
    if {"probability", "base_probability"} <= set(data):
        if not np.isfinite(data[["probability", "base_probability"]].to_numpy(dtype=float)).all():
            raise ValueError("Probabilities must be finite")
        data["probability_gain"] = data.probability - data.base_probability
        metrics.append("probability_gain")
    tables = []
    for key, setting in data.groupby(keys, sort=True):
        correct = setting[setting.base_correct.astype(bool)]
        prompt_scores = correct.groupby(["prompt", "mode"])[metrics].mean().reset_index()
        for metric in metrics:
            scores = prompt_scores.rename(columns={"prompt": "unit", "mode": "arm", metric: "value"})
            scores = scores.assign(cluster=scores.unit, stratum="all")
            table = paired_cluster_bootstrap(
                scores, arms=arms, comparisons=[("J", "logit"), ("J", "random")],
                **bootstrap_options,
            )
            tables.append(table.assign(
                **dict(zip(keys, key, strict=True)), metric=metric,
                n_trials=correct.trial_id.nunique(),
                n_eligible_prompts=setting.prompt.nunique(),
                aggregation="target mean within prompt; equal prompt mean",
            ))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


#: Datasets with multihop-easy grouping: phrasings -> base items -> families.
EASY_DATASETS = ("multihop-easy", "multihop-easy-fewshot")


def retrieval_bootstrap(
    words: pd.DataFrame,
    *,
    ks: Sequence[int] = (1, 2, 5, 10, 20, 50, 100),
    layers: slice | Sequence[int] = slice(None, -1),
    easy_metadata: pd.DataFrame | None = None,
    **bootstrap_options,
) -> pd.DataFrame:
    """Bootstrap pass@k and normalized log-k AUC from paired cached rank vectors.

    Preserve the benchmark's word -> item averaging. For multihop-easy and its
    few-shot copy (:data:`EASY_DATASETS`, which share item names), average
    phrasings within base items, then base items within families, then families
    equally. Resample ``group_id`` clusters within families, preserving related
    base items as well as their phrasings. Coverage is conditional on supported
    annotations. Neither layers nor k are chosen from the observed results.
    """
    ks = list(ks)
    if len(ks) < 2 or any(not isinstance(k, int) or k < 1 for k in ks) or ks != sorted(set(ks)):
        raise ValueError("ks must be at least two distinct increasing positive integers")
    arms = ["J-lens", "logit lens"]
    word_keys = ["dataset", "item", "kind", "word", "role"]
    if missing := {*word_keys, "lens", "ranks"} - set(words):
        raise ValueError(f"Missing rank columns: {sorted(missing)}")
    data = words[words.lens.isin(arms)].copy()
    if data[word_keys + ["lens"]].isna().any().any():
        raise ValueError("Rank identities cannot be null")
    if data.duplicated([*word_keys, "lens"]).any():
        raise ValueError("Duplicate word/lens ranks; select a single span/subset first")
    if not data.groupby(word_keys).lens.nunique().eq(2).all():
        raise ValueError("Every word must have both paired lenses")

    def best_rank(ranks):
        if ranks is None or (np.isscalar(ranks) and pd.isna(ranks)):
            return np.nan
        array = np.asarray(ranks, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all() or (array < 1).any():
            raise ValueError("Expected finite positive one-dimensional rank vectors")
        selected = array[layers]
        if not selected.size:
            raise ValueError("Selected layer scope is empty")
        return selected.min()

    data["best"] = data.ranks.map(best_rank)
    support = data.assign(supported=data.best.notna()).pivot(
        index=word_keys, columns="lens", values="supported",
    )
    if not support[arms[0]].eq(support[arms[1]]).all():
        raise ValueError("Paired lenses have different annotation support")
    tables = []
    for (dataset, kind), group in data.groupby(["dataset", "kind"], sort=True):
        hits = (group.best.to_numpy()[:, None] <= np.asarray(ks)).astype(float)
        hits[group.best.isna()] = np.nan
        values = np.column_stack([hits, np.trapezoid(hits, np.log(ks), axis=1) / np.log(ks[-1] / ks[0])])
        metrics = [("pass_at_k", k) for k in ks] + [("log_k_auc", None)]
        for j, (metric, k) in enumerate(metrics):
            scored = group.assign(value=values[:, j])
            prompt_scores = scored.groupby(["item", "lens"]).value.mean().reset_index()
            if dataset in EASY_DATASETS:
                required_metadata = ["item", "base_id", "group_id", "family"]
                if easy_metadata is None or not set(required_metadata) <= set(easy_metadata):
                    raise ValueError("multihop-easy requires base/group/family metadata")
                metadata = easy_metadata[required_metadata]
                if metadata.item.duplicated().any() or metadata.isna().any().any():
                    raise ValueError("Easy metadata must be non-null and unique by item")
                merged = prompt_scores.merge(metadata, on="item", how="left", validate="many_to_one")
                if merged[required_metadata].isna().any().any():
                    raise ValueError("Missing multihop-easy grouping metadata")
                if (merged.groupby("base_id")[["group_id", "family"]].nunique() > 1).any().any():
                    raise ValueError("Each base item must belong to one group/family")
                scores = merged.groupby(["base_id", "group_id", "family", "lens"]).value.mean().reset_index()
                scores = scores.rename(columns={
                    "base_id": "unit", "group_id": "cluster", "family": "stratum", "lens": "arm",
                })
                aggregation = "word -> phrasing -> base -> family -> equal-family mean"
            else:
                scores = prompt_scores.rename(columns={"item": "unit", "lens": "arm"})
                scores = scores.assign(cluster=scores.unit, stratum="all")
                aggregation = "word mean within item; equal item mean"
            table = paired_cluster_bootstrap(
                scores, arms=arms, comparisons=[("J-lens", "logit lens")],
                **bootstrap_options,
            )
            first = group[group.lens.eq(arms[0])]
            tables.append(table.assign(
                dataset=dataset, kind=kind, metric=metric, k=k,
                n_words=len(first), n_words_scored=int(first.best.notna().sum()),
                n_items=first.item.nunique(),
                n_items_scored=first.loc[first.best.notna(), "item"].nunique(),
                aggregation=aggregation,
            ))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def introspection_bootstrap(
    results: pd.DataFrame,
    *,
    keys: Sequence[str] = ("span", "band", "strength"),
    arms: Sequence[str] = ("J", "logit", "random"),
    comparisons: Sequence[tuple[str, str]] = (
        ("J", "logit"), ("J", "random"), ("J", "control"),
    ),
    **bootstrap_options,
) -> pd.DataFrame:
    """Concept-clustered intervals for injected-thought report rates.

    ``results`` has one row per concept and setting, with columns ``concept``,
    ``mode`` (``J``, ``logit``, ``random``), ``rank`` (the concept's next-token
    rank at the readout), and ``keys``; ``strength`` 0 rows are the shared
    no-injection control. Each injected setting is paired, concept by concept,
    with the three direction arms and the control of the same span and band,
    so a contrast against ``control`` is the effect of injecting at all.
    Metrics are ``named`` (rank 1), ``top5`` (rank <= 5) and ``rr`` (1/rank);
    intervals are for their means over concepts (a median has none here).
    Extra ``keys`` such as ``origin`` split the concepts into separate tables;
    ``arms`` and ``comparisons`` name other injected vectors (``control`` is
    always added as an arm and may appear in comparisons).
    """
    keys = list(keys)
    arms = list(arms)
    required = {*keys, "concept", "mode", "rank"}
    if "strength" not in keys:
        raise ValueError("keys must include strength")
    if missing := required - set(results):
        raise ValueError(f"Missing introspection columns: {sorted(missing)}")
    data = results.loc[results["mode"].isin(arms), list(required)].copy()
    if data.isna().any().any():
        raise ValueError("Introspection identities and ranks cannot be null")
    if not (data["rank"] >= 1).all():
        raise ValueError("Ranks must be positive")
    if data.duplicated([*keys, "concept", "mode"]).any():
        raise ValueError("Duplicate concept/setting/arm")
    if not data.groupby([*keys, "concept"])["mode"].nunique().eq(len(arms)).all():
        raise ValueError("Every concept must have all paired arms")
    data["named"] = data["rank"].eq(1).astype(float)
    data["top5"] = data["rank"].le(5).astype(float)
    data["rr"] = 1.0 / data["rank"]
    metrics = ["named", "top5", "rr"]

    setting = [k for k in keys if k != "strength"]
    control = data[data.strength.eq(0)]
    if (control.groupby([*setting, "concept"])[metrics].nunique() > 1).any().any():
        raise ValueError("The strength-0 control must not depend on the arm")
    control = control.drop_duplicates([*setting, "concept"]).assign(mode="control")
    injected = data[data.strength.ne(0)]
    tables = []
    for key, group in injected.groupby(keys, sort=True):
        key = dict(zip(keys, key, strict=True))
        baseline = control
        for column in setting:
            baseline = baseline[baseline[column].eq(key[column])]
        paired = pd.concat([group, baseline[baseline.concept.isin(group.concept)]])
        if paired.concept.nunique() != baseline.concept.nunique():
            raise ValueError(f"Control and injected concepts differ at {key}")
        for metric in metrics:
            scores = paired.rename(columns={"concept": "unit", "mode": "arm", metric: "value"})
            scores = scores.assign(cluster=scores.unit, stratum="all")
            table = paired_cluster_bootstrap(
                scores, arms=[*arms, "control"], comparisons=list(comparisons),
                **bootstrap_options,
            )
            tables.append(table.assign(
                **key, metric=metric, aggregation="equal concept mean",
            ))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
