"""Execute the convergence analysis cell on tiny cached ranks, without a model."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jlens.readout_cache import item_scores


@pytest.fixture
def analysis(tmp_path):
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/gpt2-xl/jacobian_lens/fit_convergence_ci.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    source = next(
        "".join(cell["source"]) for cell in cells
        if "def score_frame(" in "".join(cell["source"])
    )
    # Two targets in item 0 distinguish word coverage from item coverage.
    words = pd.DataFrame({
        "dataset": ["mixed"] * 3 + ["zero"] * 2 + ["mixed"] * 2,
        "kind": ["target"] * 5 + ["intermediate"] * 2,
        "item_index": [0, 0, 1, 2, 3, 0, 1],
        "ids": [[1], [], [], [], [], [2], [3]],
    })
    snapshot = np.array([[1, 2]] + [[np.nan, np.nan]] * 4 + [[1, 2]] * 2)
    logit = snapshot.copy()
    logit[0] = [2, 2]
    meta = pd.DataFrame(
        [("order", 134, 0)]
        + [("boot", 134, i) for i in range(2)]
        + [("half", 67, i) for i in range(4)]
        + [("perm", n, 0) for n in (32, 64, 96, 112)],
        columns=["experiment", "N", "rep"],
    )
    env = {
        "np": np, "pd": pd, "item_scores": item_scores, "words": words,
        "DATASETS": ["mixed", "zero"], "R_snapshot": snapshot, "R_logit": logit,
        "R_lenses": np.stack([snapshot] * len(meta)), "lens_meta": meta,
        "N_SNAPSHOT": 134, "ITEM_BOOT": 1000, "rng": np.random.default_rng(0),
        "RUN_DIR": tmp_path,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        exec(compile(source, str(path), "exec"), env)
    return env


@pytest.mark.parametrize("k", [1, 10, 100, None])
def test_bootstrap_resamples_only_supported_paired_items(analysis, k):
    draws = analysis["eval_bootstrap"]("target", k)
    j, logit, n, difference = draws["mixed"]
    assert n == 1
    expected_j = 0 if k is None else 1
    expected_l = np.log10(2) if k is None else int(k >= 2)
    np.testing.assert_array_equal(j, np.full(1000, expected_j))
    np.testing.assert_array_equal(logit, np.full(1000, expected_l))
    assert difference == expected_j - expected_l
    j, logit, n, difference = draws["zero"]
    assert n == 0
    assert np.isnan(j).all() and np.isnan(logit).all() and np.isnan(difference)


def test_bootstrap_requires_support_in_both_lenses(analysis):
    analysis["R_logit"][0] = np.nan
    j, logit, n, difference = analysis["eval_bootstrap"]("target", 1)["mixed"]
    assert n == 0
    assert np.isnan(j).all() and np.isnan(logit).all() and np.isnan(difference)


def test_summary_and_export_include_per_dataset_coverage(analysis):
    exported = pd.read_csv(analysis["RUN_DIR"] / "summary.csv")
    for frame in (analysis["curves"], analysis["summary"], exported):
        for dataset, total, supported, items in [("mixed", 3, 1, 1), ("zero", 2, 0, 0)]:
            rows = frame[frame.dataset == dataset]
            assert (rows.n_targets_total == total).all()
            assert (rows.n_targets_supported == supported).all()
            assert (rows.n_items_total == 2).all()
            assert (rows.n_target_items_supported == items).all()
            targets = rows[rows.kind == "target"]
            assert (targets.n_score_items_supported == items).all()
            if "n_paired_items_supported" in targets:
                assert (targets.n_paired_items_supported == items).all()
        probes = frame[frame.kind == "intermediate"]
        assert (probes.n_score_items_supported == 2).all()
        assert (probes.n_target_items_supported == 1).all()
    summary = analysis["summary"]
    mixed = summary[(summary.dataset == "mixed") & (summary.kind == "target")]
    assert (mixed["eval sd"] == 0).all()
    assert np.isfinite(mixed[["eval CI lo", "eval CI hi", "J − logit CI lo",
                              "J − logit CI hi", "fit sd (boot)"]]).all().all()
    zero = summary[summary.dataset == "zero"]
    estimates = [column for column in summary if column not in ("dataset", "kind", "metric")
                 and not column.startswith("n_")]
    assert zero[estimates].isna().all().all()


def test_finite_statistics_handle_partial_and_zero_support(analysis):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert analysis["supported_sd"]([1, np.nan, 3]) == np.std([1, 3], ddof=1)
        assert analysis["supported_percentile"]([1, np.nan, 3], 50) == 2
        for data in ([], [np.nan, np.nan]):
            assert np.isnan(analysis["supported_mean"](data))
            assert np.isnan(analysis["supported_sd"](data))
            assert np.isnan(analysis["supported_percentile"](data, 50))
