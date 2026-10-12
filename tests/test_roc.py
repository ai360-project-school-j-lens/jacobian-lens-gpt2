"""Regression checks for intermediate-vs-control ROC-AUC on cached rank vectors."""

import numpy as np
import pandas as pd
import pytest

from jlens.evaluation import pass_at_k
from jlens.roc import (
    intermediate_control_roc,
    roc_bootstrap,
    roc_curve,
    weighted_roc_auc,
)


def words_frame(rows):
    """rows: (dataset, item, kind, word, lens, ranks-or-None)."""
    return pd.DataFrame([
        dict(dataset=d, item=i, kind=k, word=w, role=0, lens=lens,
             ranks=None if r is None else np.asarray(r, dtype=float))
        for d, i, k, w, lens, r in rows
    ])


def both_lenses(rows, j_scale=1):
    return words_frame(
        [(d, i, k, w, "logit lens", r) for d, i, k, w, r in rows]
        + [(d, i, k, w, "J-lens", None if r is None else [x * j_scale for x in r])
           for d, i, k, w, r in rows]
    )


def brute_force_auc(pos, pos_w, neg, neg_w):
    total = sum(wp * wn * (1.0 if p < n else 0.5 if p == n else 0.0)
                for p, wp in zip(pos, pos_w, strict=True)
                for n, wn in zip(neg, neg_w, strict=True))
    return total / (sum(pos_w) * sum(neg_w))


def test_weighted_auc_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(1, 6, 9).astype(float), rng.integers(1, 6, 7).astype(float)
    pos_w, neg_w = rng.random(9), rng.random(7)
    assert weighted_roc_auc(pos, pos_w, neg, neg_w) == pytest.approx(
        brute_force_auc(pos, pos_w, neg, neg_w))
    assert np.isnan(weighted_roc_auc(pos, pos_w, np.array([]), np.array([])))


def test_roc_points_are_pass_at_k_and_auc_uses_item_weights():
    rows = [
        ("d", "a", "intermediate", "x", [9, 1, 50]),
        ("d", "a", "intermediate", "y", [30, 40, 50]),
        ("d", "a", "control", "c", [7, 8, 9]),
        ("d", "b", "intermediate", "z", [3, 2, 50]),
        ("d", "b", "control", "e", [100, 200, 300]),
        ("d", "b", "control", "f", None),  # unsupported: excluded, not a miss
        ("d", "c", "control", "g", [1, 4, 4]),
    ]
    words = both_lenses(rows)
    layers = slice(None, -1)
    curve = roc_curve(words, layers)
    expected = pass_at_k(words, [1, 2, 7, 30, 100], layers)
    for k in [1, 2, 7, 30, 100]:
        point = curve[curve.lens.eq("J-lens") & curve.k.le(k)].iloc[-1]
        rates = expected[expected.lens.eq("J-lens") & expected.k.eq(k)].set_index("kind").score
        assert point.tpr == pytest.approx(rates["intermediate"])
        assert point.fpr == pytest.approx(rates["control"])
    # Intermediates x=1, y=30 (weight 1/4 each), z=2 (1/2); controls 7, 100, 1 (1/3 each).
    expected_auc = brute_force_auc([1, 30, 2], [0.25, 0.25, 0.5], [7, 100, 1], [1, 1, 1])
    table = intermediate_control_roc(words, layers)
    assert table.roc_auc.tolist() == pytest.approx([expected_auc] * 2)
    assert table.n_control_words_scored.tolist() == [3, 3]
    # The trapezoidal area under the full curve equals the Mann-Whitney AUC.
    j_curve = curve[curve.lens.eq("J-lens")]
    assert np.trapezoid(j_curve.tpr, j_curve.fpr) == pytest.approx(expected_auc)


def test_multihop_easy_families_have_equal_weight_and_missing_controls_are_nan():
    rows = [
        ("multihop-easy", "a1", "intermediate", "w", [1, 9]),
        ("multihop-easy", "a2", "intermediate", "w", [3, 9]),
        ("multihop-easy", "a1", "control", "q", [2, 9]),
        ("multihop-easy", "b1", "intermediate", "v", [10, 9]),
        ("multihop-easy", "b1", "control", "u", [5, 9]),
        ("multihop", "m", "intermediate", "t", [1, 9]),
    ]
    metadata = pd.DataFrame({
        "item": ["a1", "a2", "b1"], "base_id": ["A", "A", "B"],
        "group_id": ["g1", "g1", "g2"], "family": ["geo", "geo", "math"],
    })
    table = intermediate_control_roc(both_lenses(rows), [0], metadata).set_index(["dataset", "lens"])
    # Positives: base A (phrasings 1 and 3, weight 1/2 each) and base B (10, weight 1).
    expected = brute_force_auc([1, 3, 10], [0.5, 0.5, 1], [2, 5], [1, 1])
    assert table.loc[("multihop-easy", "J-lens"), "roc_auc"] == pytest.approx(expected)
    assert np.isnan(table.loc[("multihop", "J-lens"), "roc_auc"])


def test_paired_bootstrap_identical_lenses_is_degenerate_and_reproducible():
    rows = [("d", f"i{n}", kind, f"{kind}{n}", [rank, 99])
            for n, (a, b) in enumerate([(1, 5), (2, 3), (8, 4), (3, 9), (6, 2), (1, 7)])
            for kind, rank in [("intermediate", a), ("control", b)]]
    same = roc_bootstrap(both_lenses(rows), [0], n_resamples=300, seed=1)
    difference = same[same.statistic.eq("difference")].iloc[0]
    assert difference.estimate == 0 and difference.status == "degenerate"
    arm = same[same.statistic.eq("arm")].iloc[0]
    assert arm.status == "ok" and arm.ci_low <= arm.estimate <= arm.ci_high
    first = roc_bootstrap(both_lenses(rows), [0], n_resamples=300, seed=1)
    pd.testing.assert_frame_equal(same, first)
