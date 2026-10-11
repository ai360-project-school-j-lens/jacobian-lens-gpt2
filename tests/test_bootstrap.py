"""Statistical regression checks using explicit finite prompt populations."""

import numpy as np
import pandas as pd
import pytest

from jlens.bootstrap import (
    paired_cluster_bootstrap,
    retrieval_bootstrap,
    swap_bootstrap,
)
from jlens.evaluation import pass_at_k


def paired(values, *, clusters=None, strata=None):
    return pd.DataFrame([
        dict(unit=str(i), cluster=str(i) if clusters is None else clusters[i],
             stratum="all" if strata is None else strata[i], arm=arm, value=value)
        for i, row in enumerate(values)
        for arm, value in zip(["J", "logit"], row, strict=True)
    ])


def bootstrap(frame, **kwargs):
    return paired_cluster_bootstrap(
        frame, arms=["J", "logit"], comparisons=[("J", "logit")],
        n_resamples=4000, **kwargs,
    )


def test_pairing_cancels_shared_prompt_difficulty():
    result = bootstrap(paired([(0, 0), (1, 1), (0, 0), (1, 1)]))
    assert (result.loc[:1, "status"] == "ok").all()
    difference = result.iloc[2]
    assert difference.estimate == 0
    assert difference.status == "degenerate"
    assert pd.isna(difference.ci_low)


def test_cluster_multiplicity_matches_enumerated_two_cluster_population():
    # Three units belong to one cluster, and one to the other. The bootstrap
    # means are 0, 3/4, 1 with probabilities 1/4, 1/2, 1/4 respectively.
    result = bootstrap(paired(
        [(1, 0), (1, 0), (1, 0), (0, 0)], clusters=["a", "a", "a", "b"],
    )).iloc[0]
    distribution = np.array([0, 0.75, 0.75, 1])
    assert result.estimate == 0.75
    assert result.n_units == 4 and result.n_clusters == 2
    assert result.ci_low == 0 and result.ci_high == 1
    assert result.standard_error == pytest.approx(distribution.std(), abs=0.015)


def test_equal_family_weight_and_reproducible_memory_batching():
    frame = paired([(0, 0)] * 8 + [(1, 0)] * 2, strata=["a"] * 8 + ["b"] * 2)
    first = bootstrap(frame, seed=11, batch_size=17)
    second = bootstrap(frame, seed=11, batch_size=501)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].estimate == 0.5  # Pooled-unit mean would be 0.2.
    assert first.iloc[0].status == "degenerate"


def test_missing_support_is_counted_and_not_converted_to_failure():
    result = bootstrap(paired([(1, 0), (0, 1), (np.nan, np.nan)]))
    assert (result.n_units_total == 3).all()
    assert (result.n_units == 2).all()
    assert result.iloc[0].estimate == 0.5
    unsupported = bootstrap(paired([(np.nan, np.nan)]))
    assert (unsupported.status == "no_supported_units").all()
    assert unsupported.estimate.isna().all()


@pytest.mark.parametrize("values,status", [
    ([(0, 0), (0, 0)], "degenerate"),
    ([(1, 0)], "too_few_clusters"),
])
def test_no_false_precision_for_constant_controls_or_singletons(values, status):
    result = bootstrap(paired(values))
    assert (result.status == status).all()
    assert result[["ci_low", "ci_high", "standard_error"]].isna().all().all()


@pytest.mark.parametrize("change,message", [
    (lambda f: f.iloc[:-1], "all paired arms"),
    (lambda f: pd.concat([f, f.iloc[:1]]), "Duplicate"),
    (lambda f: f.assign(value=[1, np.nan, 0, 0]), "different support"),
    (lambda f: f.assign(value=[1, np.inf, 0, 0]), "finite"),
    (lambda f: f.assign(cluster="same", stratum=["a", "a", "b", "b"]), "nested"),
])
def test_bad_pairing_and_cluster_metadata_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        bootstrap(change(paired([(1, 0), (0, 0)])))


def swaps(repeats=1):
    return pd.DataFrame([
        dict(dataset="toy", band="middle", alpha=1, trial_id=f"{prompt}-{target}",
             prompt=str(prompt), mode=arm, base_correct=prompt != 4,
             success=(prompt + target) % 3 == 0 if arm == "J" else False,
             changed=arm != "random", probability=0.2, base_probability=0.1)
        for prompt in range(5) for target in range(3 * repeats)
        for arm in ["J", "logit", "random"]
    ])


def test_swap_targets_are_averaged_before_resampling_and_filtering_is_paired():
    frame = swaps()
    result = swap_bootstrap(frame, n_resamples=1000)
    score = result[result.metric.eq("success") & result.statistic.eq("arm") & result.arm.eq("J")].iloc[0]
    assert score.estimate == pytest.approx(1 / 3)
    assert score.n_units == 4 and score.n_trials == 12
    assert score.n_eligible_prompts == 5
    repeated = swap_bootstrap(swaps(3), n_resamples=1000)
    pd.testing.assert_frame_equal(result.drop(columns="n_trials"), repeated.drop(columns="n_trials"))
    bad = frame.copy()
    bad.loc[0, "base_correct"] = False
    with pytest.raises(ValueError, match="baseline"):
        swap_bootstrap(bad)
    with pytest.raises(ValueError, match="all paired arms"):
        swap_bootstrap(frame.iloc[1:])


def rank_words():
    rows = []
    for item, pairs in enumerate([
        [(1, 9), (9, 1)], [(1, 9)], [(None, None)], [(9, 9)],
    ]):
        for role, (j, logit) in enumerate(pairs):
            for arm, value in [("J-lens", j), ("logit lens", logit)]:
                rows.append(dict(dataset="toy", item=str(item), word=str(role), role=role,
                                 kind="intermediate", lens=arm,
                                 ranks=None if value is None else np.array([value, 3, 1])))
    return pd.DataFrame(rows)


def test_retrieval_matches_existing_item_weighting_and_auc():
    words = rank_words()
    result = retrieval_bootstrap(words, ks=[1, 3, 10], n_resamples=1000)
    expected = pass_at_k(words, [1, 3, 10], slice(None, -1))
    for row in expected.itertuples():
        actual = result[result.arm.eq(row.lens) & result.statistic.eq("arm")
                        & result.metric.eq("pass_at_k") & result.k.eq(row.k)].iloc[0]
        assert actual.estimate == row.score
        assert actual.n_words == row.n_words and actual.n_words_scored == row.n_words_scored
        assert actual.n_units == row.n_items_scored
    for arm in ["J-lens", "logit lens"]:
        values = expected[expected.lens.eq(arm)].sort_values("k")
        auc = np.trapezoid(values.score, np.log(values.k)) / np.log(10)
        actual = result[result.arm.eq(arm) & result.statistic.eq("arm") & result.metric.eq("log_k_auc")].iloc[0]
        assert actual.estimate == pytest.approx(auc)
    final = retrieval_bootstrap(words, ks=[1, 10], layers=[-1], n_resamples=100)
    assert final[final.statistic.eq("difference")].estimate.eq(0).all()


def test_easy_phrasings_bases_groups_and_equal_family_weights():
    words, metadata = [], []
    for family, bases in [("a", [0, 0, 1, 1]), ("b", [1, 1])]:
        for base, hit in enumerate(bases):
            for phrasing in range(2):
                item = f"{family}-{base}-{phrasing}"
                metadata.append(dict(item=item, family=family, base_id=f"{family}-{base}",
                                     group_id=f"{family}-{base // 2 if family == 'a' else base}"))
                for arm in ["J-lens", "logit lens"]:
                    words.append(dict(dataset="multihop-easy", item=item, word="x", role=0,
                                      kind="intermediate", lens=arm,
                                      ranks=np.array([1 if hit and arm == "J-lens" else 10, 10])))
    result = retrieval_bootstrap(pd.DataFrame(words), ks=[1, 10],
                                 easy_metadata=pd.DataFrame(metadata), n_resamples=1000)
    score = result[result.statistic.eq("arm") & result.arm.eq("J-lens") & result.k.eq(1)].iloc[0]
    assert score.estimate == 0.75
    assert score.n_items == 12 and score.n_units == 6 and score.n_clusters == 4
    assert score.ci_low == 0.5 and score.ci_high == 1
    assert score.n_strata == 2


def test_retrieval_rejects_mismatched_word_support_before_item_averaging():
    words = rank_words()
    words.at[0, "ranks"] = None
    with pytest.raises(ValueError, match="annotation support"):
        retrieval_bootstrap(words, ks=[1, 10])
