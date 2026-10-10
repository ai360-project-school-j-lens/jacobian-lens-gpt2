"""Run generic notebook setup cells offline: no weights, corpora, or Drive downloads."""

import json
import sys
from copy import copy, deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import nbformat
import pandas as pd
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import jlens
from jlens.hf import Layout
from jlens.notebook_setup import (
    legacy_final_candidates,
    persistence_root,
    runtime_metadata,
)
from tests.test_generic_evaluation import NativeTokenizer

NOTEBOOK = (Path(__file__).resolve().parents[1]
            / "notebooks/model_agnostic/model_agnostic_lens_dataset.ipynb")
CELLS = {c.id: c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells}


def run(cell, ns, replacements=()):
    source = CELLS[cell]
    for old, new in replacements:
        assert old in source
        source = source.replace(old, new)
    exec(compile(source, f"{NOTEBOOK.name}:{cell}", "exec"), ns)


@pytest.fixture
def namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    hf = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=129, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
    ))
    hf.config._name_or_path = "offline/qwen"
    model = jlens.from_hf(hf, NativeTokenizer(False))
    lens = jlens.JacobianLens({0: torch.eye(16) * 1.0001}, d_model=16, n_prompts=3)
    ns = {
        "model": model, "hf_model": hf, "torch": torch, "Path": Path,
        "MODEL_ID": "offline/qwen", "DTYPE": torch.float32, "DEVICE": "cpu",
        "REPO_DIR": tmp_path / "repo", "FIT_MIX": {"web": 12},
        "display": lambda *args: None,
        "load_fit_prompts": Mock(return_value=pd.DataFrame({
            "source": ["web"] * 4, "text": ["short", "m" * 30, "l" * 180, "q" * 50],
        })),
        "fit_with_progress": Mock(return_value=lens),
        "benchmark_dim_batches": Mock(return_value=[
            {"dim_batch": 4, "status": "ok", "seconds": 9.0},
            {"dim_batch": 8, "status": "ok", "seconds": 3.0},
            {"dim_batch": 16, "status": "ok", "seconds": 5.0},
            {"dim_batch": 32, "status": "OOM", "seconds": None},
        ]),
    }
    run("generic-012", ns)
    return ns


def assert_no_work(ns):
    for key in ("load_fit_prompts", "fit_with_progress", "benchmark_dim_batches"):
        ns[key].assert_not_called()


def reset_work(ns):
    for key in ("load_fit_prompts", "fit_with_progress", "benchmark_dim_batches"):
        ns[key].reset_mock()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_generic_model_fastest_batch_corpus_and_fp32_save(namespace, dtype):
    ns = namespace
    ns["hf_model"].to(dtype=dtype)
    ns["DTYPE"] = dtype
    run("generic-012", ns)
    assert ns["PERSIST_ROOT"] == Path.home() / "jacobian-lens-runs"
    assert f"offline--qwen_{str(dtype).removeprefix('torch.')}_seq128_dataset-v1_" in ns["RUN_DIR"].name
    run("generic-fit-run", ns)
    fit = ns["fit_with_progress"].call_args
    assert fit.args[0] is ns["model"]
    assert fit.kwargs == dict(dim_batch=8, max_seq_len=128, skip_first=16,
                             checkpoint_path=str(ns["CHECKPOINT_PATH"]),
                             checkpoint_every=5, resume=True)
    texts = ns["load_fit_prompts"].return_value.text.tolist()
    assert fit.args[1] == texts
    assert [call.args[1] for call in ns["benchmark_dim_batches"].call_args_list] == [texts[2], texts[3]]
    assert all(call.args[0] is ns["model"] for call in ns["benchmark_dim_batches"].call_args_list)
    assert [r["text"] for r in json.loads(ns["PROMPTS_PATH"].read_text())] == texts
    assert json.loads(ns["BENCHMARK_PATH"].read_text())["dim_batch"] == 8
    saved = torch.load(ns["LENS_PATH"], weights_only=True)
    assert saved["J"][0].dtype == torch.float32
    assert torch.equal(saved["J"][0], ns["fit_with_progress"].return_value.jacobians[0])
    assert {p.dtype for p in ns["hf_model"].parameters()} == {dtype}
    assert ns["DIM_BATCH"] is None
    reset_work(ns)
    run("generic-fit-run", ns)
    assert_no_work(ns)


@pytest.mark.parametrize("stage", ["fresh", "resume", "final"])
@pytest.mark.parametrize("change", [
    "MODEL_ID", "DTYPE", "DEVICE", "MAX_SEQ_LEN", "FIT_FRACTION", "FIT_SEED",
    "DIM_BATCH", "DIM_BATCH_CANDIDATES", "CHECKPOINT_EVERY", "RUN_LABEL",
    "MEMORY_HEADROOM_GIB", "MEMORY_HEADROOM_FRACTION", "FIT_MIX",
    "precision", "dtype", "partial_dtype", "model", "hf_model", "tokenizer", "revision",
])
def test_stale_runtime_rejected_before_work(namespace, monkeypatch, stage, change):
    ns = namespace
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        run("generic-012", ns)
        if stage != "fresh":
            run("generic-fit-run", ns)
            if stage == "resume":
                ns["LENS_PATH"].unlink()
                ns["CHECKPOINT_PATH"].touch()
        if change == "precision":
            torch.set_float32_matmul_precision("high")
        elif change == "dtype":
            ns["hf_model"].to(torch.bfloat16)
        elif change == "partial_dtype":
            # Include a parameter outside decoder blocks in validation.
            ns["hf_model"].lm_head.to(torch.bfloat16)
        elif change in ("model", "hf_model"):
            ns[change] = SimpleNamespace(**vars(ns[change]))
        elif change == "tokenizer":
            ns["model"].tokenizer = NativeTokenizer(False)
        elif change == "revision":
            ns["hf_model"].config._commit_hash = "changed"
        else:
            ns[change] = {
                "MODEL_ID": "other/model", "DTYPE": torch.bfloat16, "DEVICE": "meta",
                "MAX_SEQ_LEN": 64, "FIT_FRACTION": 0.5, "FIT_SEED": 1,
                "DIM_BATCH": 2, "DIM_BATCH_CANDIDATES": (2, 4), "CHECKPOINT_EVERY": 10,
                "RUN_LABEL": "other", "MEMORY_HEADROOM_GIB": 3,
                "MEMORY_HEADROOM_FRACTION": 0.2, "FIT_MIX": {"web": 100},
            }[change]
        loader = Mock()
        monkeypatch.setattr(jlens.JacobianLens, "load", loader)
        reset_work(ns)
        with pytest.raises(RuntimeError, match="rerun configuration cell"):
            run("generic-fit-run", ns)
        loader.assert_not_called()
        assert_no_work(ns)
        assert ns["fitted_lens"] is None
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("stage", ["resume", "final"])
@pytest.mark.parametrize("damage", ["corpus", "manifest", "order", "config", "malformed"])
def test_provenance_damage_rejected(namespace, stage, damage):
    ns = namespace
    run("generic-fit-run", ns)
    if stage == "resume":
        ns["LENS_PATH"].unlink()
        ns["CHECKPOINT_PATH"].touch()
    if damage in ("corpus", "manifest"):
        ns["PROMPTS_PATH" if damage == "corpus" else "MANIFEST_PATH"].unlink()
    elif damage == "order":
        rows = json.loads(ns["PROMPTS_PATH"].read_text())
        ns["PROMPTS_PATH"].write_text(json.dumps(rows[::-1]))
    elif damage == "config":
        saved = json.loads(ns["MANIFEST_PATH"].read_text())
        saved["config"]["dtype"] = "other"
        ns["MANIFEST_PATH"].write_text(json.dumps(saved))
    else:
        ns["MANIFEST_PATH"].write_text("{bad")
    reset_work(ns)
    with pytest.raises((RuntimeError, ValueError)):
        run("generic-fit-run", ns)
    assert_no_work(ns)
    assert ns["fitted_lens"] is None


def test_resume_reuses_corpus_rebenchmarks_and_manual_override(namespace):
    ns = namespace
    run("generic-fit-run", ns)
    ns["LENS_PATH"].unlink()
    ns["CHECKPOINT_PATH"].touch()
    reset_work(ns)
    ns["load_fit_prompts"].side_effect = AssertionError("must not redownload")
    ns["benchmark_dim_batches"].return_value[2]["seconds"] = 1.0
    run("generic-fit-run", ns)
    ns["load_fit_prompts"].assert_not_called()
    assert ns["fit_with_progress"].call_args.kwargs["dim_batch"] == 16
    # Intentional new configuration must get a new directory.
    old = ns["RUN_DIR"]
    run("generic-012", ns, [("DIM_BATCH = None", "DIM_BATCH = 2")])
    assert ns["RUN_DIR"] != old
    reset_work(ns)
    ns["load_fit_prompts"].side_effect = None
    run("generic-fit-run", ns)
    ns["benchmark_dim_batches"].assert_not_called()
    assert ns["fit_with_progress"].call_args.kwargs["dim_batch"] == 2


@pytest.mark.parametrize("location", ["repo", "drive_flat", "drive_full", "drive_nested"])
def test_legacy_discovery_stops_refit_and_prints_opt_in(namespace, tmp_path, monkeypatch, capsys, location):
    ns = namespace
    drive = tmp_path / "drive"
    full = ns["MODEL_ID"].replace("/", "__")
    path = {
        "repo": Path(ns["REPO_DIR"]) / "lens_checkpoints" / full / "lens.pt",
        "drive_flat": drive / "jacobian-lens/qwen_jacobian_lens_dataset.pt",
        "drive_full": drive / f"jacobian-lens/{full}_jacobian_lens_dataset.pt",
        "drive_nested": drive / "jacobian-lens" / full / "lens.pt",
    }[location]
    path.parent.mkdir(parents=True)
    ns["fit_with_progress"].return_value.save(str(path))
    # Real discovery helper with an offline stand-in for the mounted Drive root.
    ns["legacy_final_candidates"] = lambda repo, name: legacy_final_candidates(repo, name, drive)
    with pytest.raises(RuntimeError, match="Legacy final found"):
        run("generic-fit-run", ns)
    assert_no_work(ns)
    assert not ns["RUN_DIR"].exists()
    assert f"EXISTING_LENS_PATH = Path({str(path)!r})" in capsys.readouterr().out
    ns["EXISTING_LENS_PATH"] = path
    run("generic-fit-run", ns)
    assert_no_work(ns)
    assert "UNVERIFIED legacy lens" in capsys.readouterr().out
    assert ns["fitted_lens"].d_model == ns["model"].d_model
    ns["EXISTING_LENS_PATH"] = None
    ns["ALLOW_NEW_FIT_WITH_LEGACY"] = True
    run("generic-fit-run", ns)
    ns["fit_with_progress"].assert_called_once()


def test_missing_legacy_or_checkpoint_override_never_refits(namespace):
    ns = namespace
    path = Path.home() / "previous.pt"
    ns["EXISTING_LENS_PATH"] = path
    with pytest.raises(FileNotFoundError):
        run("generic-fit-run", ns)
    torch.save({"checkpoint": True}, path)
    with pytest.raises(ValueError, match="not a JacobianLens"):
        run("generic-fit-run", ns)
    assert_no_work(ns)
    assert ns["fitted_lens"] is None


def test_legacy_discovery_does_not_match_checkpoints_or_other_models(namespace):
    ns = namespace
    directory = Path(ns["REPO_DIR"]) / "lens_checkpoints"
    (directory / "offline__qwen").mkdir(parents=True)
    (directory / "other__model").mkdir()
    (directory / "offline__qwen/lens_ckpt.pt").touch()
    (directory / "other__model/lens.pt").touch()
    assert legacy_final_candidates(ns["REPO_DIR"], ns["MODEL_ID"], Path.home()) == []


def test_cuda_headroom_and_no_safe_fallback(namespace, monkeypatch):
    ns = namespace
    ns["DEVICE"] = "cuda:0"
    placement = SimpleNamespace(dtype=torch.float32, device=torch.device("cuda:0"))
    monkeypatch.setattr(ns["hf_model"], "parameters", lambda: iter([placement]))
    for layer in ns["model"].layers:
        monkeypatch.setattr(layer, "parameters", lambda: iter([placement]))
    run("generic-012", ns)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(12 * 2**30, 20 * 2**30)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", Mock(return_value=2 * 2**30))
    rows = ns["benchmark_dim_batches"].return_value
    for row, peak in zip(rows, (8, 12, 10, 20), strict=True):
        row["peak_reserved_GiB"] = peak
    run("generic-fit-run", ns)
    assert ns["fit_with_progress"].call_args.kwargs["dim_batch"] == 16
    assert json.loads(ns["BENCHMARK_PATH"].read_text())["memory_budget_GiB"] == 11
    ns["LENS_PATH"].unlink()
    reset_work(ns)
    for row in rows:
        row["seconds"] = float("nan")
    with pytest.raises(RuntimeError, match="No measured safe batch"):
        run("generic-fit-run", ns)
    ns["fit_with_progress"].assert_not_called()


def test_colab_mount_failure_local_opt_out(namespace, monkeypatch):
    ns = namespace
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=Mock())
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    real_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: str(p) == "/content/drive/MyDrive" or real_is_dir(p))
    run("generic-012", ns)
    colab.drive.mount.assert_called_once_with("/content/drive")
    assert ns["RUN_DIR"].is_relative_to("/content/drive/MyDrive/jacobian-lens-runs")
    colab.drive.mount.side_effect = RuntimeError("mount failed")
    with pytest.raises(RuntimeError, match="mount failed"):
        run("generic-012", ns)
    assert persistence_root(False, Path.home()) == Path.home()


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("failure", [None, "evaluation", "assertion"])
def test_identity_scoped_restoration(namespace, precision, failure):
    ns = namespace
    previous = torch.get_float32_matmul_precision()
    previous_cudnn = torch.backends.cudnn.allow_tf32
    ns.update(evals={"order-ops": [{}]}, pd=pd, identity_lens=Mock(), handwritten_logit_lens=Mock())

    def evaluate(*args, **kwargs):
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.is_autocast_enabled("cpu")
        assert kwargs["layer_logit_readout"] is ns["handwritten_logit_lens"]
        if failure == "evaluation":
            raise RuntimeError("evaluation failed")
        frame = pd.DataFrame({"lens": ["logit lens", "J-lens"],
                              "metric": [1.0, 2.0 if failure == "assertion" else 1.0]})
        return frame, frame.copy()

    ns["evaluate_paired"] = evaluate
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            if failure is None:
                run("generic-014", ns)
            else:
                with pytest.raises(RuntimeError if failure == "evaluation" else AssertionError):
                    run("generic-014", ns)
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_float32_matmul_precision() == precision
            assert torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def test_fit_rejects_autocast_before_download(namespace):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="outside autocast"):
            run("generic-fit-run", namespace)
        assert torch.is_autocast_enabled("cpu")
    assert_no_work(namespace)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_tiny_generic_calibration_fit_and_checkpoint_resume(namespace, dtype):
    from jlens.evaluation import fit_with_progress
    from jlens.sharded_fitting import benchmark_dim_batches

    ns = namespace
    ns["hf_model"].to(dtype=dtype)
    ns["DTYPE"] = dtype
    ns["load_fit_prompts"].return_value = pd.DataFrame({
        "source": ["synthetic", "synthetic"], "text": ["a" * 19, "b" * 22],
    })
    ns["benchmark_dim_batches"] = benchmark_dim_batches
    ns["fit_with_progress"] = fit_with_progress
    run("generic-012", ns, [
        ("DIM_BATCH_CANDIDATES = (4, 8, 16, 32)", "DIM_BATCH_CANDIDATES = (4, 8)"),
        ("MAX_SEQ_LEN = 128", "MAX_SEQ_LEN = 24"),
        ("CHECKPOINT_EVERY = 5", "CHECKPOINT_EVERY = 1"),
    ])
    run("generic-fit-run", ns)
    original = ns["fitted_lens"].jacobians[0].clone()
    assert ns["fitted_lens"].n_prompts == 2
    assert ns["CHECKPOINT_PATH"].is_file()
    ns["LENS_PATH"].unlink()  # Exercise the real fitter's completed-checkpoint resume.
    ns["load_fit_prompts"].reset_mock()
    run("generic-fit-run", ns)
    ns["load_fit_prompts"].assert_not_called()
    assert torch.equal(ns["fitted_lens"].jacobians[0], original)
    assert {p.dtype for p in ns["hf_model"].parameters()} == {dtype}


def test_adapter_must_match_hf_model_even_after_config_rerun(namespace):
    ns = namespace
    ns["hf_model"] = Qwen2ForCausalLM(ns["hf_model"].config)
    with pytest.raises(RuntimeError, match="different weights"):
        run("generic-012", ns)
    assert_no_work(ns)


@pytest.mark.parametrize("stage", ["fresh", "resume", "final"])
@pytest.mark.parametrize("component", [
    "embed_tokens", "norm", "lm_head", "layers", "decoder", "missing_head",
    "detached_bf16_head",
])
def test_replaced_adapter_components_rejected_before_work(namespace, monkeypatch, stage, component):
    ns = namespace
    if stage != "fresh":
        run("generic-fit-run", ns)
        if stage == "resume":
            ns["LENS_PATH"].unlink()
            ns["CHECKPOINT_PATH"].touch()
    hf = ns["hf_model"]
    if component == "detached_bf16_head":
        hf.lm_head = deepcopy(hf.lm_head)
        ns["model"]._lm_head.to(torch.bfloat16)
    elif component == "missing_head":
        del hf.lm_head
    elif component == "decoder":
        hf.model = copy(hf.model)
    else:
        parent = hf if component == "lm_head" else hf.model
        # Keep all parameters identical: only module identity detects staleness.
        setattr(parent, component, copy(getattr(parent, component)))
    assert {p.dtype for p in hf.parameters()} == {torch.float32}
    reset_work(ns)
    loader = Mock()
    monkeypatch.setattr(jlens.JacobianLens, "load", loader)
    with pytest.raises(RuntimeError, match="rebuild from_hf"):
        run("generic-fit-run", ns)
    assert ns["fitted_lens"] is None
    # Rerunning configuration must not bless a stale adapter, either.
    with pytest.raises(RuntimeError, match="rebuild from_hf"):
        run("generic-012", ns)
    loader.assert_not_called()
    assert_no_work(ns)
    if component != "missing_head":
        ns["model"] = jlens.from_hf(hf, NativeTokenizer(False))
        run("generic-012", ns)


@pytest.mark.parametrize("shared", ["blocks", "decoder", "all_components"])
def test_different_hf_root_sharing_decoder_rejected_on_configuration(namespace, shared):
    ns = namespace
    original = ns["hf_model"]
    other = Qwen2ForCausalLM(original.config)
    if shared == "blocks":
        other.model.layers = original.model.layers
    else:
        other.model = original.model
    if shared == "all_components":
        other.lm_head = original.lm_head
    other.eval()
    ns["hf_model"] = other
    assert all(a is b for a, b in zip(ns["model"].layers, other.model.layers, strict=True))
    with pytest.raises(RuntimeError, match="rebuild from_hf"):
        run("generic-012", ns)
    assert_no_work(ns)


@pytest.mark.parametrize("component", ["embed", "norm", "lm_head", "layers", "path"])
def test_runtime_metadata_supports_and_validates_custom_layout(component):
    from tests.test_hf_layout import _make_hf_mock

    layout = Layout("custom.decoder", layers="blocks", norm="final_norm",
                    embed="tokens", lm_head="readout")
    hf = _make_hf_mock(layout)
    model = jlens.from_hf(hf, NativeTokenizer(False), layout=layout)
    metadata = runtime_metadata(model, hf, "offline/custom", torch.float32, "cpu")
    assert metadata["d_model"] == 8
    assert metadata["n_layers"] == 2
    if component == "path":
        hf.custom.decoder = copy(hf.custom.decoder)
    else:
        parent = hf if component == "lm_head" else hf.custom.decoder
        name = getattr(layout, component)
        setattr(parent, name, copy(getattr(parent, name)))
    with pytest.raises(RuntimeError, match="rebuild from_hf"):
        runtime_metadata(model, hf, "offline/custom", torch.float32, "cpu")


def test_runtime_metadata_rejects_unverifiable_generic_adapter(namespace):
    ns = namespace
    generic = SimpleNamespace(**vars(ns["model"]))
    with pytest.raises(RuntimeError, match="requires HFLensModel/from_hf"):
        runtime_metadata(generic, ns["hf_model"], ns["MODEL_ID"], ns["DTYPE"], ns["DEVICE"])


def test_notebook_valid_and_outputs_clear():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert all(c.execution_count is None and not c.outputs
               for c in notebook.cells if c.cell_type == "code")
