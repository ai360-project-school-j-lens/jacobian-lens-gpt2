"""Statistical regression checks using explicit finite prompt populations."""

import numpy as np
import pandas as pd
import pytest

from jlens.bootstrap import (
    introspection_bootstrap,
    paired_cluster_bootstrap,
    retrieval_bootstrap,
    retrieval_bootstrap_by_layer,
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


def test_layerwise_retrieval_uses_each_block_and_preserves_coverage():
    words = rank_words()
    result = retrieval_bootstrap_by_layer(
        words, n_layers=3, ks=[1, 3, 10], n_resamples=1000,
    )
    top1 = result[result.metric.eq("pass_at_k") & result.k.eq(1)]
    arms = top1[top1.statistic.eq("arm")].pivot(
        index="layer", columns="arm", values="estimate",
    )
    np.testing.assert_allclose(arms["J-lens"], [0.5, 0, 1])
    np.testing.assert_allclose(arms["logit lens"], [1 / 6, 0, 1])
    assert result.n_units_total.eq(4).all() and result.n_units.eq(3).all()
    assert result.n_words.eq(5).all() and result.n_words_scored.eq(4).all()
    assert result.n_layers.eq(3).all()
    np.testing.assert_allclose(result.depth_fraction, result.layer / 2)
    np.testing.assert_allclose(result.depth_percent, result.layer * 50)
    assert result.is_final_layer.equals(result.layer.eq(2))
    assert result.scope.eq("single block").all()
    final = result[result.is_final_layer & result.statistic.eq("difference")]
    assert final.estimate.eq(0).all() and final.status.eq("degenerate").all()
    assert final.ci_low.isna().all() and final.ci_high.isna().all()

    # CI endpoints and AUC use the existing paired estimand at each block.
    for layer in range(3):
        expected = retrieval_bootstrap(
            words, layers=[layer], ks=[1, 3, 10], n_resamples=1000,
        )
        actual = result.loc[result.layer.eq(layer), expected.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


def test_layerwise_retrieval_retains_unsupported_layers():
    result = retrieval_bootstrap_by_layer(
        rank_words().assign(ranks=None), n_layers=3, ks=[1, 10], n_resamples=100,
    )
    assert set(result.layer) == {0, 1, 2}
    assert result.status.eq("no_supported_units").all()
    assert result[["estimate", "ci_low", "ci_high"]].isna().all().all()
    assert result.n_units_total.eq(4).all() and result.n_units.eq(0).all()


@pytest.mark.parametrize("n_layers", [1, 0, -1, 3.0, True])
def test_layerwise_retrieval_rejects_invalid_layer_count(n_layers):
    with pytest.raises(ValueError, match="n_layers"):
        retrieval_bootstrap_by_layer(rank_words(), n_layers=n_layers)


def test_layerwise_retrieval_rejects_truncated_rank_vectors():
    words = rank_words()
    words.at[0, "ranks"] = np.array([1, 2])
    with pytest.raises(ValueError, match="n_layers entries"):
        retrieval_bootstrap_by_layer(words, n_layers=3)


@pytest.mark.parametrize("dataset", ["multihop-easy", "multihop-easy-fewshot"])
def test_easy_phrasings_bases_groups_and_equal_family_weights(dataset):
    words, metadata = [], []
    for family, bases in [("a", [0, 0, 1, 1]), ("b", [1, 1])]:
        for base, hit in enumerate(bases):
            for phrasing in range(2):
                item = f"{family}-{base}-{phrasing}"
                metadata.append(dict(item=item, family=family, base_id=f"{family}-{base}",
                                     group_id=f"{family}-{base // 2 if family == 'a' else base}"))
                for arm in ["J-lens", "logit lens"]:
                    words.append(dict(dataset=dataset, item=item, word="x", role=0,
                                      kind="intermediate", lens=arm,
                                      ranks=np.array([1 if hit and arm == "J-lens" else 10, 10])))
    result = retrieval_bootstrap(pd.DataFrame(words), ks=[1, 10],
                                 easy_metadata=pd.DataFrame(metadata), n_resamples=1000)
    score = result[result.statistic.eq("arm") & result.arm.eq("J-lens") & result.k.eq(1)].iloc[0]
    assert score.estimate == 0.75
    assert score.n_items == 12 and score.n_units == 6 and score.n_clusters == 4
    assert score.ci_low == 0.5 and score.ci_high == 1
    assert score.n_strata == 2
    layers = retrieval_bootstrap_by_layer(
        pd.DataFrame(words), n_layers=2, ks=[1, 10],
        easy_metadata=pd.DataFrame(metadata), n_resamples=1000,
    )
    first = layers.loc[layers.layer.eq(0), result.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, result)
    last = layers[layers.layer.eq(1) & layers.k.eq(1)]
    assert last.estimate.eq(0).all() and last.status.eq("degenerate").all()


def test_retrieval_rejects_mismatched_word_support_before_item_averaging():
    words = rank_words()
    words.at[0, "ranks"] = None
    with pytest.raises(ValueError, match="annotation support"):
        retrieval_bootstrap(words, ks=[1, 10])


def introspection_results(n_concepts=8):
    """J names every concept at strength 1, logit half of them, random none."""
    rows = []
    for c in range(n_concepts):
        origin = "original" if c % 2 else "added"
        for mode in ["J", "logit", "random"]:
            rows.append(dict(span="turn", band="1-3", strength=0.0, mode=mode,
                             concept=f"c{c}", origin=origin, rank=100 + c))
            rank = {"J": 1, "logit": 1 if c < n_concepts // 2 else 7, "random": 500}[mode]
            rows.append(dict(span="turn", band="1-3", strength=1.0, mode=mode,
                             concept=f"c{c}", origin=origin, rank=rank))
    return pd.DataFrame(rows)


def test_introspection_bootstrap_pairs_arms_with_the_shared_control():
    table = introspection_bootstrap(introspection_results(), n_resamples=500)
    assert set(table.strength) == {1.0}  # the control is an arm, not a setting
    named = table[table.metric.eq("named")].set_index(["statistic", "arm", "reference"]).estimate
    assert named["arm", "J", ""] == 1.0
    assert named["arm", "logit", ""] == 0.5
    assert named["arm", "random", ""] == 0.0
    assert named["arm", "control", ""] == 0.0
    assert named["difference", "J", "control"] == 1.0
    assert named["difference", "J", "logit"] == 0.5
    rr = table[table.metric.eq("rr") & table.statistic.eq("arm") & table.arm.eq("control")]
    assert rr.estimate.iloc[0] == pytest.approx(np.mean([1 / (100 + c) for c in range(8)]))
    status = table[table.metric.eq("named")].set_index(["statistic", "arm", "reference"]).status
    assert status["arm", "J", ""] == "degenerate"  # all named: no interval, not certainty
    assert status["difference", "J", "logit"] == "ok"


def test_introspection_bootstrap_splits_by_extra_keys():
    table = introspection_bootstrap(
        introspection_results(), keys=["span", "band", "strength", "origin"], n_resamples=200,
    )
    assert set(table.origin) == {"added", "original"}
    assert table.groupby("origin").n_units.first().eq(4).all()


def test_introspection_bootstrap_rejects_an_arm_dependent_control():
    results = introspection_results()
    results.loc[results.strength.eq(0) & results["mode"].eq("J") & results.concept.eq("c0"), "rank"] = 1
    with pytest.raises(ValueError, match="control"):
        introspection_bootstrap(results, n_resamples=200)


def test_introspection_bootstrap_rejects_a_missing_arm():
    results = introspection_results()
    results = results[~(results["mode"].eq("random") & results.concept.eq("c3"))]
    with pytest.raises(ValueError, match="paired arms"):
        introspection_bootstrap(results, n_resamples=200)


def test_introspection_bootstrap_accepts_other_arms():
    results = introspection_results().replace({"mode": {"J": "J-space", "logit": "non-J-space"}})
    table = introspection_bootstrap(
        results, arms=["J-space", "non-J-space", "random"],
        comparisons=[("J-space", "non-J-space"), ("non-J-space", "control")], n_resamples=200,
    )
    named = table[table.metric.eq("named")].set_index(["statistic", "arm", "reference"]).estimate
    assert named["difference", "J-space", "non-J-space"] == 0.5
    assert named["difference", "non-J-space", "control"] == 0.5

