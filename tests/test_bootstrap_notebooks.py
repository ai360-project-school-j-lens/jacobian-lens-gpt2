"""Exercise notebook cache paths without downloads, models, or source mutations."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer, models

from jlens.bootstrap import (
    retrieval_bootstrap,
    retrieval_bootstrap_by_layer,
    swap_bootstrap,
)
from jlens.position_benchmark import reduce_position_ranks

ROOT = Path(__file__).resolve().parents[1]
SWAP = ROOT / "notebooks/model_agnostic/token_swap/model_agnostic_token_swap_bootstrap.ipynb"
RETRIEVAL = ROOT / "notebooks/model_agnostic/latent_bridge_v2_lens_benchmark_bootstrap.ipynb"
LAYERWISE = RETRIEVAL.with_name("latent_bridge_v2_lens_benchmark_bootstrap_layerwise.ipynb")


def cells(path):
    notebook = nbformat.read(path, as_version=4)
    return {cell.id: cell.source for cell in notebook.cells}


@pytest.fixture
def forbid_models(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cache analysis must not download or evaluate a model")

    for path in [
        "transformers.AutoConfig.from_pretrained",
        "transformers.AutoTokenizer.from_pretrained",
        "transformers.AutoModelForCausalLM.from_pretrained",
        "transformers.AutoModelForCausalLM.from_config",
        "huggingface_hub.HfApi.model_info",
        "jlens.artifact_provenance.download_lens_artifact",
        "jlens.JacobianLens.load",
        "jlens.from_hf",
        "tests.tiny.TinyDecoder.forward",
    ]:
        monkeypatch.setattr(path, forbidden)
    return forbidden


@pytest.mark.parametrize("correct_baseline", [True, False])
def test_swap_cache_reanalysis_preserves_source(tmp_path, forbid_models, correct_baseline):
    source = cells(SWAP)
    state = dict(
        RUN_MODE="demo", RUN_MODEL=False, ALPHAS=[0.0, 1.0, 2.0], SEED=0,
        Path=Path, pd=pd, json=json, display=Mock(),
    )
    exec(source["cfe690d7"], state)  # Build the notebook's explicit synthetic rows.
    state["results"]["base_correct"] = correct_baseline
    cache = tmp_path / "source"
    cache.mkdir()
    results_path = cache / "results.json"
    state["results"].to_json(results_path, orient="records", double_precision=15)
    (cache / "manifest.json").write_text(json.dumps(state["metadata"]))
    before = {p.name: p.read_bytes() for p in cache.iterdir()}

    # Redirect only derived outputs while keeping the notebook's provenance code.
    analysis_root = tmp_path / "analysis"
    for path in [SWAP, ROOT / "jlens/bootstrap.py"]:
        target = analysis_root / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    state.update(
        ROOT=analysis_root, RUN_MODE="cache", CACHE_RUN_DIR=str(cache),
        swap_bootstrap=swap_bootstrap, BOOTSTRAP_OPTIONS=dict(n_resamples=100, seed=0),
        run_swap_experiment=forbid_models,
    )
    for cell in ["e712e0aa", "cfe690d7", "11aa98cb", "57a242b2", "bootstrap-swap-analysis"]:
        exec(source[cell], state)
    first_dir = state["ANALYSIS_DIR"]
    first_points = state["ci_table"].estimate.copy()
    assert state["PRIMARY_BAND"] == "middle"
    if not correct_baseline:
        assert state["ci_table"].status.eq("no_supported_units").all()
    state["BOOTSTRAP_OPTIONS"]["n_resamples"] = 200
    exec(source["bootstrap-swap-analysis"], state)
    assert state["ANALYSIS_DIR"] != first_dir
    pd.testing.assert_series_equal(first_points, state["ci_table"].estimate)
    assert {p.name: p.read_bytes() for p in cache.iterdir()} == before


def rank_cache(tmp_path, *, embedded, nested_config):
    """A complete minimal rank cache; no learned weights or forward passes."""
    evals = {
        "latent-bridge/toy": [dict(name="bridge", prompt="toy", intermediates=["x"])],
        "toy": [dict(name="ordinary", prompt="toy", intermediates=["x"])],
        "multihop-easy": [
            dict(name=f"base-{i}-{p}", prompt="toy", intermediates=["x"],
                 base_id=f"base-{i}", group_id=f"group-{i}", family="family",
                 split="test", template_id=str(p))
            for i in range(3) for p in range(2)
        ],
    }
    rows = [
        dict(dataset=dataset, item=item["name"], kind="intermediate", word="x",
             role=0, lens=arm, ranks=np.array([1 if arm == "J-lens" else 20, 2]))
        for dataset, items in evals.items() for item in items
        for arm in ["J-lens", "logit lens"]
    ]
    all_words = pd.DataFrame(rows)
    position_words = all_words[all_words.dataset.eq("latent-bridge/toy")].copy()
    position_words["positions"] = [[0] for _ in range(len(position_words))]
    position_words["ranks"] = position_words.ranks.map(lambda ranks: ranks[:, None])
    anth_words = all_words[~all_words.dataset.eq("latent-bridge/toy")].copy()
    for frame in [position_words, anth_words]:
        frame.attrs["evaluation"] = {"items": frame.item.nunique()}
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))
    config = {"vocab_size": 32, "id2label": {0: "LABEL_0", 1: "LABEL_1"}}
    manifest = dict(
        mode="pretrained", model="offline/cached-model", revision="saved-commit",
        dtype="torch.bfloat16", bos_policy="prepend", tokenizer=backend.to_str(),
        config={"text_config": config} if nested_config else config,
        evals_sha256=hashlib.sha256(json.dumps(evals, sort_keys=True).encode()).hexdigest(),
    )
    payload = dict(
        manifest=manifest, position_words=position_words, anth_words=anth_words,
        latent_items=position_words[["dataset", "item"]].drop_duplicates(),
        anth_items=anth_words[["dataset", "item"]].drop_duplicates(), n_layers=2,
        spans={("latent-bridge/toy", "bridge"): {"subject": [0]}},
    )
    if embedded:
        payload["evals"] = evals
    cache = tmp_path / "results.pkl"
    pd.to_pickle(payload, cache)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return cache, evals


def retrieval_state(cache, evals, forbidden):
    return dict(
        RUN_MODE="cache", EXISTING_RANK_CACHE=str(cache), REPO=ROOT,
        Path=Path, pd=pd, np=np, json=json, hashlib=hashlib, display=Mock(),
        load_latent_bridge=lambda _: {"latent-bridge/toy": evals["latent-bridge/toy"]},
        load_eval=lambda _, name: evals[name], DATASETS=["toy"],
        reduce_position_ranks=reduce_position_ranks, evaluate_position_ranks=forbidden,
        evaluate_paired_batched=forbidden, FORCE_RECOMPUTE=True,
        BOOTSTRAP_OPTIONS=dict(n_resamples=100, seed=0, confidence=0.95),
        K_VALUES=[1, 10], K=1, LAYERWISE_SPANS=["subject"], LAYERWISE_WORKERS=2,
        CACHE_ROOT=cache.parent / "analysis",
    )


@pytest.mark.parametrize("embedded,nested_config", [(False, False), (True, True)])
@pytest.mark.parametrize("notebook", [RETRIEVAL, LAYERWISE])
def test_retrieval_cache_only_uses_saved_identity_and_grouping(
    tmp_path, forbid_models, embedded, nested_config, notebook,
):
    cache, evals = rank_cache(tmp_path, embedded=embedded, nested_config=nested_config)
    source = cells(notebook)
    state = retrieval_state(cache, evals, forbid_models)
    before = cache.read_bytes()
    for cell in ["ea930c3e", "benchmark-lens-source", "c1eb4c15", "e435187a",
                 "bootstrap-analysis-destination"]:
        exec(source[cell], state)
    assert state["run_model_id"] == "offline/cached-model"
    assert state["run_dtype"] == "torch.bfloat16"
    assert state["BOS_POLICY"] == "prepend"
    assert state["config"].get_text_config().vocab_size == 32
    assert state["tokenizer"].decode([1]) == "x"
    result = retrieval_bootstrap(
        state["anth_words"], easy_metadata=state["easy_metadata"], ks=[1, 10],
        n_resamples=100,
    )
    assert result[result.dataset.eq("multihop-easy")].n_clusters.eq(3).all()
    first_dir = state["RUN_DIR"]
    state["BOOTSTRAP_OPTIONS"]["n_resamples"] = 200
    exec(source["bootstrap-analysis-destination"], state)
    assert state["RUN_DIR"] != first_dir
    assert cache.read_bytes() == before


@pytest.mark.parametrize("mismatch", ["sidecar", "grouping"])
@pytest.mark.parametrize("notebook", [RETRIEVAL, LAYERWISE])
def test_retrieval_cache_rejects_mismatched_provenance(
    tmp_path, forbid_models, mismatch, notebook,
):
    cache, evals = rank_cache(tmp_path, embedded=False, nested_config=False)
    if mismatch == "sidecar":
        (tmp_path / "manifest.json").write_text("{}")
        message = "sidecar manifest disagree"
    else:
        evals["multihop-easy"][0]["group_id"] = "changed-group"
        message = "metadata changed"
    with pytest.raises(ValueError, match=message):
        exec(cells(notebook)["ea930c3e"], retrieval_state(cache, evals, forbid_models))


def test_layerwise_cache_plots_and_exports_match_without_model_calls(
    tmp_path, monkeypatch, forbid_models, capsys,
):
    cache, evals = rank_cache(tmp_path, embedded=True, nested_config=True)
    payload = pd.read_pickle(cache)
    words = payload["anth_words"]
    # Make one easy base a miss; the interval now reflects three groups,
    # rather than counting its two phrasings as independent observations.
    for index in words.index[words.item.str.startswith("base-1-")]:
        words.at[index, "ranks"] = np.array([20, 2])
    pd.to_pickle(payload, cache)
    before = cache.read_bytes()
    source = cells(LAYERWISE)
    state = retrieval_state(cache, evals, forbid_models)
    for cell in ["ea930c3e", "benchmark-lens-source", "c1eb4c15", "e435187a",
                 "bootstrap-analysis-destination"]:
        exec(source[cell], state)
    state.update(
        easy_words=state["anth_words"].merge(state["easy_metadata"], on="item"),
        retrieval_bootstrap_by_layer=Mock(wraps=retrieval_bootstrap_by_layer),
        bootstrap_ci=retrieval_bootstrap(
            state["anth_words"], easy_metadata=state["easy_metadata"],
            ks=[1, 10], **state["BOOTSTRAP_OPTIONS"],
        ),
        PRINT_LAYERWISE_CSV=True, plt=plt,
        COLORS={"J-lens": "blue", "logit lens": "red"},
    )
    compute = state["retrieval_bootstrap_by_layer"]

    def interrupted_compute(*args, **kwargs):
        if compute.call_count == 2:
            raise RuntimeError("Simulated interrupted bootstrap")
        return retrieval_bootstrap_by_layer(*args, **kwargs)

    compute.side_effect = interrupted_compute
    with pytest.raises(RuntimeError, match="interrupted bootstrap"):
        exec(source["bootstrap-layerwise-analysis"], state)
    parts_dir = state["RUN_DIR"] / "layerwise_parts"
    saved_parts = {p: p.read_bytes() for p in parts_dir.glob("*.pkl")}
    assert len(saved_parts) == 1
    assert not (state["RUN_DIR"] / "layerwise_bootstrap.pkl").exists()
    compute.reset_mock(side_effect=True)
    exec(source["bootstrap-layerwise-analysis"], state)
    assert compute.call_count == len(list(parts_dir.glob("*.pkl"))) - 1
    assert {p: p.read_bytes() for p in saved_parts} == saved_parts
    original = state["layerwise_ci"].copy()
    calls = state["retrieval_bootstrap_by_layer"].call_count
    exec(source["bootstrap-layerwise-analysis"], state)
    assert state["retrieval_bootstrap_by_layer"].call_count == calls
    pd.testing.assert_frame_equal(state["layerwise_ci"], original)
    exec(source["bootstrap-layerwise-export"], state)
    exported = pd.read_csv(state["RUN_DIR"] / "layerwise_plot_data.csv")
    expected = state["layerwise_plot_data"]
    pd.testing.assert_frame_equal(
        exported, expected.replace({"": np.nan}).reset_index(drop=True),
        check_dtype=False,
    )
    assert exported.model_id.eq("offline/cached-model").all()
    assert exported.model_revision.eq("saved-commit").all()
    assert exported.dtype.eq("torch.bfloat16").all()
    assert exported.source_cache_sha256.eq(hashlib.sha256(before).hexdigest()).all()
    assert set(exported.layer) == {0, 1} and set(exported.depth_percent) == {0, 100}
    easy = exported[
        exported.dataset.eq("multihop-easy") & exported.layer.eq(0)
        & exported.statistic.eq("arm") & exported.arm.eq("J-lens")
    ].iloc[0]
    assert easy.estimate == pytest.approx(2 / 3)
    assert easy.status == "ok" and easy.ci_low == 0 and easy.ci_high == 1
    assert easy.n_units == easy.n_clusters == 3 and easy.n_items == 6
    assert "export_schema,analysis_id,model_id" in capsys.readouterr().out

    monkeypatch.setattr(plt, "show", Mock())
    previous = set(plt.get_fignums())
    try:
        exec(source["bootstrap-layerwise-plots"], state)
        assert plt.show.call_count == 2
        exec(source["bootstrap-layerwise-latent-profiles"], state)
        assert plt.show.call_count == 3
        assert (state["RUN_DIR"] / "latent_bridge_layer_profiles.png").is_file()
        for name in ["readouts", "differences"]:
            for extension in ["png", "svg"]:
                assert (state["RUN_DIR"] / f"bootstrap_layerwise_{name}.{extension}").is_file()
    finally:
        for number in set(plt.get_fignums()) - previous:
            plt.close(number)
    assert cache.read_bytes() == before
