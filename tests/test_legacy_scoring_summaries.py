"""Legacy rank summaries exclude null targets without changing probe semantics."""

import numpy as np
import pandas as pd
import pytest

from jlens.evaluation import layer_hit_rate, layer_median_rank, pass_at_k


@pytest.fixture
def words():
    return pd.DataFrame([
        {"dataset": "mixed", "kind": "target", "item": item,
         "ranks": ranks, "single_token": ranks is not None}
        for item, ranks in [
            ("a", np.array([1, 9])), ("a", np.array([9, 1])),
            ("a", None), ("b", np.array([9, 9])), ("c", None),
        ]
    ] + [{"dataset": "unsupported", "kind": "target", "item": "d",
          "ranks": None, "single_token": False}])


@pytest.mark.parametrize("layers,expected", [(None, 0.5), ([0], 0.25), ([-1], 0.25)])
def test_pass_at_k_excludes_nulls_but_preserves_item_first_weighting(words, layers, expected):
    scores = pass_at_k(words, ks=[1], layers=layers).set_index("dataset")
    assert scores.loc["mixed", "score"] == expected
    assert scores.loc["mixed", ["n_words", "n_words_scored", "n_items", "n_items_scored"]].tolist() == [5, 3, 3, 2]
    assert pd.isna(scores.loc["unsupported", "score"])
    assert scores.loc["unsupported", ["n_words", "n_words_scored", "n_items", "n_items_scored"]].tolist() == [1, 0, 1, 0]


def test_layer_summaries_use_only_ranked_words_and_omit_unsupported_datasets(words):
    hits = layer_hit_rate(words, 1)
    medians = layer_median_rank(words)
    assert hits.index.tolist() == medians.index.tolist() == ["mixed"]
    np.testing.assert_allclose(hits.loc["mixed"], [1 / 3, 1 / 3])
    np.testing.assert_array_equal(medians.loc["mixed"], [9, 9])
    for subset in (words.iloc[:0], words[words.dataset.eq("unsupported")]):
        assert layer_hit_rate(subset, 1).empty
        assert layer_median_rank(subset).empty
    empty = pass_at_k(words.iloc[:0])
    assert empty.empty
    assert {"dataset", "kind", "k", "score", "n_items_scored"} <= set(empty)


def test_legacy_prefix_probes_remain_ranked_even_without_single_token_support(words):
    probes = words.iloc[:2].assign(kind="intermediate", single_token=False)
    scores = pass_at_k(probes, ks=[1]).iloc[0]
    assert scores.score == 1
    assert scores.n_words_scored == 2
    assert scores.n_items_scored == 1
    np.testing.assert_allclose(layer_hit_rate(probes, 1).loc["mixed"], [0.5, 0.5])


def test_paired_summaries_keep_lenses_separate(words):
    paired = pd.concat([
        words.assign(lens="logit lens"), words.assign(lens="J-lens"),
    ], ignore_index=True)
    scores = pass_at_k(paired, ks=[1])
    assert len(scores) == 4
    left, right = (scores[scores.lens.eq(name)].drop(columns="lens").reset_index(drop=True)
                   for name in ("logit lens", "J-lens"))
    pd.testing.assert_frame_equal(left, right)
