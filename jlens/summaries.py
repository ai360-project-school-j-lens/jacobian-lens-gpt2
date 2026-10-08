"""Small, inference-free notebook summaries of strict whole-token rank tables.

These views complement, not replace, the Figure 52 helpers. Input rank arrays
must be ordered by block, with the shared final-model output last. Token support
comes from the evaluator; no tokenization, prefix fallback, or rank repair occurs.
See ``docs/summaries.md`` for denominators and the public notebook API.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

_KEYS = ["dataset", "item"]
_COUNT_COLUMNS = [
    "dataset", "total", "annotated", "supported", "correct", "incorrect",
    "unsupported", "unannotated", "supported_accuracy",
]
_RETRIEVAL_COLUMNS = [
    "dataset", "lens", "subset", "kind", "k", "layer_scope", "score",
    "n_items_total", "n_items_subset", "n_items_annotated", "n_items_supported",
    "n_items_used", "n_items_excluded", "n_words_total", "n_words_supported",
    "n_words_used", "n_words_excluded", "item_coverage", "word_coverage",
]


def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")


def _equal(a, b) -> bool:
    if isinstance(a, (tuple, list, np.ndarray)) or isinstance(
        b, (tuple, list, np.ndarray)
    ):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return bool(a == b)


def _dataset_names(items: pd.DataFrame, datasets) -> list:
    if isinstance(datasets, str):
        raise ValueError("datasets must be a sequence, not a string")
    return list(dict.fromkeys(items["dataset"] if datasets is None else datasets))


def _model_items(items: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate only model-level metadata, never lens-specific readouts."""
    _require(items, [*_KEYS, "target", "model_correct"])
    if items[_KEYS].isna().any().any():
        raise ValueError("dataset/item keys cannot be missing")
    if "lens" in items and items.duplicated([*_KEYS, "lens"]).any():
        raise ValueError("Duplicate dataset/item/lens rows")
    shared = [
        c for c in (
            "target", "model_correct", "target_ids", "model_top1_id", "model_top1",
            "prompt", "readout_token", "readout_position", "n_tokens",
            "n_prompt_tokens",
        ) if c in items
    ]
    rows = []
    for key, group in items.groupby(_KEYS, sort=False):
        first = group.iloc[0]
        for column in shared:
            if not all(_equal(first[column], v) for v in group[column]):
                raise ValueError(f"Inconsistent lens copies for {key}: {column}")
        annotated = pd.notna(first["target"])
        correct = first["model_correct"]
        if pd.notna(correct) and not isinstance(correct, (bool, np.bool_)):
            raise ValueError("model_correct must be boolean or missing")
        supported = pd.notna(correct)
        if supported and not annotated:
            raise ValueError(f"Unannotated item has model_correct: {key}")
        if "target_ids" in items and bool(len(first["target_ids"])) != supported:
            raise ValueError(f"target_ids/model_correct support mismatch: {key}")
        rows.append({**dict(zip(_KEYS, key, strict=True)), "target": first["target"],
                     "annotated": bool(annotated), "supported": bool(supported),
                     "correct": bool(correct) if supported else False})
    return pd.DataFrame(rows, columns=[
        *_KEYS, "target", "annotated", "supported", "correct",
    ])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def answer_counts(
    items: pd.DataFrame, datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Count annotated answer support and model-argmax correctness once per item.

    ``unsupported`` means annotated but not supported; it excludes unannotated
    items. Accuracy is correct/supported and NaN with no supported targets.
    Missing model_correct denotes unsupported/unannotated, never incorrect.
    Shared model-level metadata must agree across lens copies.
    """
    model_items = _model_items(items)
    rows = []
    for dataset in _dataset_names(items, datasets):
        group = model_items[model_items["dataset"].eq(dataset)]
        total = len(group)
        annotated = int(group["annotated"].sum())
        supported = int(group["supported"].sum())
        correct = int(group["correct"].sum())
        rows.append(dict(
            dataset=dataset, total=total, annotated=annotated, supported=supported,
            correct=correct, incorrect=supported - correct,
            unsupported=annotated - supported, unannotated=total - annotated,
            supported_accuracy=_ratio(correct, supported),
        ))
    return pd.DataFrame(rows, columns=_COUNT_COLUMNS)


def _positive_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")


def _word_table(all_words: pd.DataFrame, model_items: pd.DataFrame) -> pd.DataFrame:
    _require(all_words, [*_KEYS, "lens", "kind", "word", "single_token", "ranks"])
    if all_words[[*_KEYS, "lens", "kind"]].isna().any().any():
        raise ValueError("Word keys cannot be missing")
    if not all(isinstance(v, (bool, np.bool_)) for v in all_words["single_token"]):
        raise ValueError("single_token must contain booleans")
    identity = [*_KEYS, "lens", "kind", "role" if "role" in all_words else "word"]
    if all_words.duplicated(identity).any():
        raise ValueError("Duplicate word rows; retain role to distinguish annotations")
    frame = all_words.merge(model_items, on=_KEYS, how="left", validate="many_to_one",
                            indicator=True, suffixes=("", "_item"))
    if not frame["_merge"].eq("both").all():
        raise ValueError("Word rows reference items absent from items")
    return frame.drop(columns="_merge")


def _ranks(value) -> np.ndarray:
    ranks = np.asarray(value, dtype=float)
    if (ranks.ndim != 1 or not len(ranks) or not np.isfinite(ranks).all()
            or (ranks < 1).any() or (ranks != np.floor(ranks)).any()):
        raise ValueError("Supported words need nonempty positive integer rank arrays")
    return ranks


def retrieval_summary(
    all_words: pd.DataFrame, items: pd.DataFrame, *, kind: str = "intermediate",
    k: int = 10, layer_scope: str = "inner",
    subsets: Sequence[str] = ("all", "correct"),
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Item-first supported-word rank-hit means, with explicit coverage counts.

    All includes *every* item (no correctness/support filter on its answer).
    Correct includes only model_correct == True. For each eligible item, average
    supported-word hits, then average items equally. Unsupported words and items
    without eligible words are excluded, not scored zero. ``inner`` drops the
    last rank entry; ``all`` retains it; ``final`` uses only it.
    """
    _positive_k(k)
    if layer_scope not in {"inner", "all", "final"}:
        raise ValueError("layer_scope must be inner, all, or final")
    if kind not in {"intermediate", "target", "control"}:
        raise ValueError("kind must be intermediate, target, or control")
    if isinstance(subsets, str) or not set(subsets) <= {"all", "correct"}:
        raise ValueError("subsets must contain only all and/or correct")
    model_items = _model_items(items)
    words = _word_table(all_words, model_items)
    words = words[words["kind"].eq(kind)].copy()
    hits = []
    for row in words.itertuples():
        if not row.single_token:
            hits.append(np.nan)
            continue
        ranks = _ranks(row.ranks)
        selected = ranks[:-1] if layer_scope == "inner" else (
            ranks[-1:] if layer_scope == "final" else ranks
        )
        hits.append(float(selected.min() <= k) if len(selected) else np.nan)
    words["_hit"] = hits
    lenses = list(dict.fromkeys([
        *(items["lens"] if "lens" in items else []), *all_words["lens"],
    ]))
    rows = []
    for dataset in _dataset_names(items, datasets):
        population = model_items[model_items["dataset"].eq(dataset)]
        for lens in lenses:
            for subset in dict.fromkeys(subsets):
                selected_items = population[
                    population["correct"].eq(True)
                ] if subset == "correct" else population
                group = words[
                    words["dataset"].eq(dataset) & words["lens"].eq(lens)
                    & words["item"].isin(selected_items["item"])
                ]
                supported = group[group["single_token"].eq(True)]
                used = group[group["_hit"].notna()]
                n_items_used = used["item"].nunique()
                rows.append(dict(
                    dataset=dataset, lens=lens, subset=subset, kind=kind, k=int(k),
                    layer_scope=layer_scope,
                    score=used.groupby("item")["_hit"].mean().mean(),
                    n_items_total=len(population), n_items_subset=len(selected_items),
                    n_items_annotated=group["item"].nunique(),
                    n_items_supported=supported["item"].nunique(),
                    n_items_used=n_items_used,
                    n_items_excluded=len(selected_items) - n_items_used,
                    n_words_total=len(group), n_words_supported=len(supported),
                    n_words_used=len(used), n_words_excluded=len(group) - len(used),
                    item_coverage=_ratio(n_items_used, len(selected_items)),
                    word_coverage=_ratio(len(used), len(group)),
                ))
    return pd.DataFrame(rows, columns=_RETRIEVAL_COLUMNS)


def target_final_counts(
    all_words: pd.DataFrame, items: pd.DataFrame, *, k: int = 10,
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Shared final-output strict-rank hits @1/@k for the annotated answer target.

    One target per annotated item, deduplicated across lenses. Final ranks and
    support must agree across copies. This is NOT retrieval of the final prompt
    token, nor rank of the model's own prediction. Tied rank-1 hits need not equal
    argmax correctness. Unsupported/no-target datasets have NaN hit rates.
    """
    _positive_k(k)
    model_items = _model_items(items)
    words = _word_table(all_words, model_items)
    targets = words[words["kind"].eq("target")]
    if targets.duplicated([*_KEYS, "lens"]).any():
        raise ValueError("Expected one annotated answer target per item/lens")
    final = {}
    for key, group in targets.groupby(_KEYS, sort=False):
        if not group["annotated"].all():
            raise ValueError(f"Target rows for an unannotated item: {key}")
        if not group["single_token"].eq(group["supported"]).all():
            raise ValueError(f"Target word/item support mismatch: {key}")
        if not all(_equal(w, t) for w, t in zip(
            group["word"], group["target"], strict=True,
        )):
            raise ValueError(f"Target word/item annotation mismatch: {key}")
        values = [
            _ranks(row.ranks)[-1] if row.single_token else np.nan
            for row in group.itertuples()
        ]
        if not all(_equal(values[0], v) for v in values):
            raise ValueError(f"Inconsistent shared final target ranks: {key}")
        final[key] = values[0]
    expected = set(model_items.loc[model_items["annotated"].eq(True), _KEYS]
                   .itertuples(index=False, name=None))
    if expected != set(final):
        raise ValueError("Need target word rows for every annotated item")
    counts = answer_counts(items, datasets=datasets)
    rows = []
    for count in counts.to_dict("records"):
        ranks = np.asarray([
            rank for (dataset, _), rank in final.items() if dataset == count["dataset"]
        ])
        for cutoff in dict.fromkeys((1, int(k))):
            hits = int((ranks <= cutoff).sum())
            rows.append({**count, "k": cutoff, "hits": hits,
                         "misses": count["supported"] - hits,
                         "hit_rate": _ratio(hits, count["supported"])})
    return pd.DataFrame(rows, columns=[
        *_COUNT_COLUMNS, "k", "hits", "misses", "hit_rate",
    ])


def plot_retrieval_summary(summary: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Compact all/correct panels; NaN bars are labelled N/A, never zero-filled."""
    _require(summary, ["dataset", "lens", "subset", "score", "kind", "k",
                       "layer_scope"])
    if summary.empty:
        raise ValueError("No retrieval summary rows to plot")
    for column in ("kind", "k", "layer_scope"):
        if summary[column].nunique() != 1:
            raise ValueError(f"Select one {column} before plotting")
    if summary.duplicated(["dataset", "lens", "subset"]).any():
        raise ValueError("Duplicate dataset/lens/subset summaries")
    datasets = list(summary["dataset"].unique())
    lenses = list(summary["lens"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    width = 0.8 / len(lenses)
    x = np.arange(len(datasets))
    for ax, subset in zip(axes, ("all", "correct"), strict=True):
        for i, lens in enumerate(lenses):
            group = summary[summary["lens"].eq(lens) & summary["subset"].eq(subset)]
            scores = group.set_index("dataset")["score"].reindex(datasets)
            positions = x - 0.4 + width * (i + 0.5)
            ax.bar(positions, scores, width, label=lens, color=f"C{i}")
            for position, value in zip(positions, scores, strict=True):
                if pd.isna(value):
                    ax.text(position, 0.02, "N/A", ha="center", fontsize=7,
                            rotation=90)
        ax.set_xticks(x, datasets, rotation=25, ha="right")
        ax.set(title="All items" if subset == "all" else "Model-correct items only",
               ylim=(0, 1), ylabel="Item-first supported-word hit rate")
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    row = summary.iloc[0]
    label = "Annotated answer target" if row["kind"] == "target" else row["kind"].title()
    scope = {"inner": "inner layers (shared final excluded)",
             "all": "all layers", "final": "shared final output"}[row["layer_scope"]]
    fig.suptitle(f"{label} retrieval @{row['k']} — {scope}")
    fig.tight_layout()
    return fig, axes


def plot_target_final_counts(counts: pd.DataFrame) -> tuple[Figure, np.ndarray]:
    """Plot shared final target hit counts (one bar per cutoff, not per lens)."""
    _require(counts, ["dataset", "k", "hits", "supported"])
    if counts.empty or counts.duplicated(["dataset", "k"]).any():
        raise ValueError("Need one nonempty final-count row per dataset/k")
    datasets = list(counts["dataset"].unique())
    cutoffs = sorted(counts["k"].unique())
    fig, axes = plt.subplots(1, 1, figsize=(8, 3.8), squeeze=False)
    ax = axes[0, 0]
    width = 0.8 / len(cutoffs)
    x = np.arange(len(datasets))
    for i, cutoff in enumerate(cutoffs):
        group = counts[counts["k"].eq(cutoff)].set_index("dataset").reindex(datasets)
        positions = x - 0.4 + width * (i + 0.5)
        eligible = group["supported"].gt(0)
        ax.bar(positions, group["hits"].where(eligible), width, label=f"@{cutoff}")
        for position, (_, row) in zip(positions, group.iterrows(), strict=True):
            label = (f"{int(row.hits)}/{int(row.supported)}"
                     if pd.notna(row.supported) and row.supported > 0 else "N/A")
            ax.annotate(label, (position, row.hits if label != "N/A" else 0),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8)
    ax.set_xticks(x, datasets, rotation=25, ha="right")
    ax.set(ylabel="Target hits / supported items", ylim=(0, None),
           title="Annotated answer target — shared final-model output\n"
                 "Strict-rank hits, not model-argmax accuracy")
    ax.margins(y=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, axes
