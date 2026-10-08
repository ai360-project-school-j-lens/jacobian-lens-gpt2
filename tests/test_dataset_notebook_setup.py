"""Execute notebook persistence/calibration cells offline, without Drive or downloads."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

import jlens

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/jacobian_lens/jacobian_logit_lens_dataset.ipynb"
)
CELLS = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}


def run(cell, namespace, replacements=()):
    source = CELLS[cell]
    for old, new in replacements:
        assert old in source
        source = source.replace(old, new)
    exec(compile(source, f"{NOTEBOOK.name}:{cell}", "exec"), namespace)


@pytest.fixture
def namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    layer = torch.nn.Linear(4, 4)
    gpt = SimpleNamespace(
        layers=[layer], n_layers=3, d_model=4,
        encode=Mock(side_effect=lambda text, max_length: torch.zeros(
            1, min(len(text) + 1, max_length), dtype=torch.long,
        )),
    )
    lens = jlens.JacobianLens(
        {0: torch.eye(4), 1: torch.eye(4)}, n_prompts=3, d_model=4,
    )
    ns = {
        "gpt": gpt, "torch": torch, "pd": pd, "np": np, "jlens": jlens,
        "display": lambda *args: None, "MODEL_ID": "offline/tiny-gpt2",
        "FIT_MIX": {"web": 12}, "LAST": 2,
        "MODEL_DTYPE": torch.float32, "DEVICE": "cpu",
        "load_fit_prompts": Mock(return_value=pd.DataFrame({
            "source": ["web"] * 4,
            "text": ["short", "m" * 30, "l" * 180, "q" * 50],
        })),
        "fit_with_progress": Mock(return_value=lens),
        "benchmark_dim_batches": Mock(return_value=[
            {"dim_batch": 4, "status": "ok", "seconds": 9.0},
            {"dim_batch": 8, "status": "ok", "seconds": 3.0},
            {"dim_batch": 16, "status": "ok", "seconds": 5.0},
            {"dim_batch": 32, "status": "OOM", "seconds": None},
        ]),
    }
    run("paired-008", ns)
    return ns


def test_local_paths_fastest_batch_and_fp32_save(namespace, capsys):
    ns = namespace
    run("paired-008", ns)
    assert ns["PERSIST_ROOT"] == Path.home() / "jacobian-lens-runs"
    run_name = ns["RUN_DIR"].name
    assert "offline--tiny-gpt2_float32_seq128_dataset-v1_" in run_name
    assert len(run_name.rsplit("_", 1)[-1]) == 12
    printed = capsys.readouterr().out
    for key in ("LENS_PATH", "CHECKPOINT_PATH", "PROMPTS_PATH", "MANIFEST_PATH"):
        assert ns[key].parent == ns["RUN_DIR"]
        assert str(ns[key]) in printed
    run("paired-fit-run", ns)
    assert ns["SELECTED_DIM_BATCH"] == 8  # Not largest or first successful.
    assert ns["DIM_BATCH"] is None  # Configured request is not overwritten.
    fit = ns["fit_with_progress"].call_args
    assert fit.kwargs["dim_batch"] == 8
    assert fit.kwargs["max_seq_len"] == 128
    assert fit.kwargs["checkpoint_path"] == str(ns["CHECKPOINT_PATH"])
    assert fit.kwargs["resume"] is True
    assert fit.kwargs["skip_first"] == 16
    texts = ns["load_fit_prompts"].return_value.text.tolist()
    assert fit.args[1] == texts  # Persist all prompts in original order, even skipped short ones.
    assert [call.args[1] for call in ns["benchmark_dim_batches"].call_args_list] == [
        texts[2], texts[3],  # Longest and median valid truncated lengths, not first/short prompt.
    ]
    assert all(call.kwargs["max_seq_len"] == 128
               for call in ns["benchmark_dim_batches"].call_args_list)
    assert ns["LENS_PATH"].is_file()
    saved = torch.load(ns["LENS_PATH"], weights_only=True)
    assert all(matrix.dtype == torch.float32 for matrix in saved["J"].values())
    assert json.loads(ns["BENCHMARK_PATH"].read_text())["dim_batch"] == 8
    assert [row["text"] for row in json.loads(ns["PROMPTS_PATH"].read_text())] == texts


def test_saved_final_skips_all_expensive_work(namespace, capsys):
    ns = namespace
    run("paired-fit-run", ns)
    for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
        ns[name].reset_mock()
    run("paired-fit-run", ns)
    for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
        ns[name].assert_not_called()
    assert "Loading verified final lens" in capsys.readouterr().out
    assert ns["PROMPTS_PATH"].exists()


@pytest.mark.parametrize("damage", [
    "missing_manifest", "missing_corpus", "model", "dtype", "precision", "order",
    "malformed_manifest", "malformed_corpus",
])
def test_managed_final_requires_compatible_provenance(namespace, monkeypatch, damage):
    ns = namespace
    run("paired-fit-run", ns)
    if damage.startswith("missing"):
        ns["MANIFEST_PATH" if damage == "missing_manifest" else "PROMPTS_PATH"].unlink()
    elif damage == "order":
        records = json.loads(ns["PROMPTS_PATH"].read_text())
        ns["PROMPTS_PATH"].write_text(json.dumps(records[::-1]))
    elif damage.startswith("malformed"):
        ns["MANIFEST_PATH" if damage == "malformed_manifest" else "PROMPTS_PATH"].write_text("{bad")
    else:
        manifest = json.loads(ns["MANIFEST_PATH"].read_text())
        key = {"model": "model_id", "dtype": "dtype", "precision": "matmul_precision"}[damage]
        manifest["config"][key] = "incompatible"
        ns["MANIFEST_PATH"].write_text(json.dumps(manifest))
    loader = Mock()
    monkeypatch.setattr(jlens.JacobianLens, "load", loader)
    for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
        ns[name].reset_mock()
    with pytest.raises((ValueError, RuntimeError)):
        run("paired-fit-run", ns)
    loader.assert_not_called()
    for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
        ns[name].assert_not_called()
    assert ns["fitted_lens"] is None


def test_explicit_legacy_final_needs_no_sidecars_or_refit(namespace, capsys):
    ns = namespace
    legacy = Path.home() / "previous-lens.pt"
    ns["fit_with_progress"].return_value.save(str(legacy))
    ns["EXISTING_LENS_PATH"] = legacy
    run("paired-fit-run", ns)
    for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
        ns[name].assert_not_called()
    assert "UNVERIFIED legacy lens" in capsys.readouterr().out
    assert not ns["RUN_DIR"].exists()


@pytest.mark.parametrize("stage", ["fresh", "resume", "final"])
@pytest.mark.parametrize("change", [
    "precision", "dtype", "partial_dtype", "device", "model", "MODEL_ID",
    "MODEL_DTYPE", "DEVICE", "MAX_SEQ_LEN", "FIT_FRACTION", "FIT_SEED",
    "DIM_BATCH", "DIM_BATCH_CANDIDATES", "CHECKPOINT_EVERY", "RUN_LABEL",
    "MEMORY_HEADROOM_GIB", "MEMORY_HEADROOM_FRACTION", "FIT_MIX", "cudnn_tf32",
])
def test_stale_runtime_rejected_before_managed_work(namespace, monkeypatch, stage, change):
    ns = namespace
    old_precision = torch.get_float32_matmul_precision()
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        run("paired-008", ns)
        if stage != "fresh":
            run("paired-fit-run", ns)
            if stage == "resume":
                ns["LENS_PATH"].unlink()
                ns["CHECKPOINT_PATH"].touch()
        if change == "precision":
            torch.set_float32_matmul_precision("high")
        elif change == "cudnn_tf32":
            torch.backends.cudnn.allow_tf32 = not old_cudnn_tf32
        elif change == "dtype":
            ns["gpt"].layers[0].to(torch.float64)
        elif change == "partial_dtype":
            layer = ns["gpt"].layers[0]
            layer.bias = torch.nn.Parameter(layer.bias.to(torch.float64))
        elif change == "device":
            ns["gpt"].layers[0].to("meta")
        elif change == "model":
            ns["gpt"] = SimpleNamespace(**vars(ns["gpt"]))
        else:
            ns[change] = {
                "MODEL_ID": "different/model", "MODEL_DTYPE": torch.float64,
                "DEVICE": "meta", "MAX_SEQ_LEN": 64, "FIT_FRACTION": 0.5,
                "FIT_SEED": 1, "DIM_BATCH": 2, "DIM_BATCH_CANDIDATES": (2, 4),
                "CHECKPOINT_EVERY": 10, "RUN_LABEL": "changed",
                "MEMORY_HEADROOM_GIB": 3, "MEMORY_HEADROOM_FRACTION": 0.2,
                "FIT_MIX": {"web": 100},
            }[change]
        loader = Mock()
        monkeypatch.setattr(jlens.JacobianLens, "load", loader)
        for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
            ns[name].reset_mock()
        with pytest.raises(RuntimeError, match="rerun configuration cell"):
            run("paired-fit-run", ns)
        loader.assert_not_called()
        for name in ("load_fit_prompts", "benchmark_dim_batches", "fit_with_progress"):
            ns[name].assert_not_called()
        assert ns["fitted_lens"] is None
        if stage == "fresh":
            assert not ns["RUN_DIR"].exists()
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def test_model_metadata_validation_does_not_read_weights(namespace, monkeypatch):
    ns = namespace
    model = SimpleNamespace(
        config=SimpleNamespace(_name_or_path=ns["MODEL_ID"]),
        parameters=Mock(return_value=iter([
            SimpleNamespace(dtype=torch.float32, device=torch.device("cpu")),
        ])),
    )
    ns["gpt"]._hf_model = model
    assert ns["current_fit_config"]() == ns["FIT_CONFIG"]
    model.config._name_or_path = "other/model"
    model.parameters.return_value = iter([
        SimpleNamespace(dtype=torch.float32, device=torch.device("cpu")),
    ])
    with pytest.raises(RuntimeError, match="Loaded model does not match"):
        ns["current_fit_config"]()


def test_resume_reuses_ordered_corpus_and_recalibrates(namespace):
    ns = namespace
    run("paired-fit-run", ns)
    ns["LENS_PATH"].unlink()
    ns["CHECKPOINT_PATH"].touch()  # The fitter is mocked, but its resume path is real.
    ns["load_fit_prompts"].reset_mock()
    ns["load_fit_prompts"].side_effect = AssertionError("must not resample on resume")
    ns["benchmark_dim_batches"].return_value[2]["seconds"] = 1.0
    run("paired-fit-run", ns)
    ns["load_fit_prompts"].assert_not_called()
    assert ns["fit_with_progress"].call_args.kwargs["dim_batch"] == 16
    assert ns["SELECTED_DIM_BATCH"] == 16
    assert ns["DIM_BATCH"] is None


@pytest.mark.parametrize("missing", ["PROMPTS_PATH", "MANIFEST_PATH"])
def test_resume_refuses_missing_sidecars(namespace, missing):
    ns = namespace
    run("paired-fit-run", ns)
    ns["LENS_PATH"].unlink()
    ns["CHECKPOINT_PATH"].touch()
    ns[missing].unlink()
    ns["fit_with_progress"].reset_mock()
    with pytest.raises(RuntimeError, match="lacks corpus/manifest"):
        run("paired-fit-run", ns)
    ns["fit_with_progress"].assert_not_called()
    assert ns["fitted_lens"] is None


def test_changed_ordered_corpus_is_rejected(namespace):
    ns = namespace
    run("paired-fit-run", ns)
    ns["LENS_PATH"].unlink()
    records = json.loads(ns["PROMPTS_PATH"].read_text())
    ns["PROMPTS_PATH"].write_text(json.dumps(records[::-1]))
    with pytest.raises(ValueError, match="manifest/corpus mismatch"):
        run("paired-fit-run", ns)


def test_manual_batch_is_used_without_benchmark(namespace):
    ns = namespace
    run("paired-008", ns, [("DIM_BATCH = None", "DIM_BATCH = 2")])
    run("paired-fit-run", ns)
    ns["benchmark_dim_batches"].assert_not_called()
    assert ns["fit_with_progress"].call_args.kwargs["dim_batch"] == 2


def test_configuration_changes_get_distinct_paths(namespace):
    ns = namespace
    original = ns["LENS_PATH"]
    for replacements in (
        [("MAX_SEQ_LEN = 128", "MAX_SEQ_LEN = 64")],
        [("FIT_FRACTION = 0.25", "FIT_FRACTION = 0.5")],
        [("DIM_BATCH = None", "DIM_BATCH = 2")],
        [('RUN_LABEL = "dataset-v1"', 'RUN_LABEL = "new-corpus"')],
    ):
        run("paired-008", ns, replacements)
        assert ns["LENS_PATH"] != original
    ns["gpt"].layers[0].to(torch.float64)
    ns["MODEL_DTYPE"] = torch.float64
    run("paired-008", ns)
    assert "float64" in ns["RUN_DIR"].name
    assert ns["LENS_PATH"] != original


def test_cuda_memory_budget_rejects_faster_unsafe_candidate(namespace, monkeypatch):
    ns = namespace
    # Mock placement metadata, not only the captured device (which is now validated).
    ns["DEVICE"] = "cuda:0"
    monkeypatch.setattr(ns["gpt"].layers[0], "parameters", lambda: iter([
        SimpleNamespace(dtype=torch.float32, device=torch.device("cuda:0")),
    ]))
    run("paired-008", ns)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(12 * 2**30, 20 * 2**30)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", Mock(return_value=2 * 2**30))
    rows = ns["benchmark_dim_batches"].return_value
    for row, peak in zip(rows, (8, 12, 10, 20), strict=True):
        row["peak_reserved_GiB"] = peak
    run("paired-fit-run", ns)
    # Budget = 12 free + 2 already reserved - max(2, .15*20) = 11 GiB.
    assert ns["memory_budget_gib"] == 11
    assert ns["SELECTED_DIM_BATCH"] == 16  # Batch 8 is faster but needs 12 GiB.
    assert ns["fit_with_progress"].call_args.kwargs["dim_batch"] == 16


@pytest.mark.parametrize("status,seconds", [("OOM", None), ("ok", float("nan"))])
def test_no_safe_batch_stops_before_fit(namespace, status, seconds):
    ns = namespace
    for row in ns["benchmark_dim_batches"].return_value:
        row.update(status=status, seconds=seconds)
    with pytest.raises(RuntimeError, match="No measured safe batch"):
        run("paired-fit-run", ns)
    ns["fit_with_progress"].assert_not_called()
    assert not ns["LENS_PATH"].exists()


def test_colab_persistence_uses_mocked_drive(namespace, monkeypatch):
    ns = namespace
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=Mock())
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    real_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda path: (
        str(path) == "/content/drive/MyDrive" or real_is_dir(path)
    ))
    run("paired-008", ns)
    colab.drive.mount.assert_called_once_with("/content/drive")
    assert ns["RUN_DIR"].is_relative_to("/content/drive/MyDrive/jacobian-lens-runs")
    colab.drive.mount.side_effect = RuntimeError("mount failed")
    with pytest.raises(RuntimeError, match="mount failed"):
        run("paired-008", ns)


def test_explicit_existing_lens_typo_never_starts_fit(namespace):
    ns = namespace
    ns["EXISTING_LENS_PATH"] = ns["LENS_PATH"]  # Does not exist.
    with pytest.raises(FileNotFoundError, match="Requested existing final lens"):
        run("paired-fit-run", ns)
    ns["load_fit_prompts"].assert_not_called()
    ns["fit_with_progress"].assert_not_called()


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("failure", [None, "evaluation", "assertion"])
def test_identity_restores_precision_and_autocast(namespace, precision, failure):
    ns = namespace
    old_precision = torch.get_float32_matmul_precision()
    old_cudnn = torch.backends.cudnn.allow_tf32
    ns["evals"] = {"order-ops": [{"prompt": "offline"}]}
    ns["identity_lens"] = Mock(return_value=object())

    def evaluate(*args, **kwargs):
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.is_autocast_enabled("cpu")
        if failure == "evaluation":
            raise RuntimeError("evaluation failed")
        frame = pd.DataFrame({
            "lens": ["logit lens", "J-lens"],
            "metric": [1.0, 2.0 if failure == "assertion" else 1.0],
        })
        return frame, frame.copy()

    ns["evaluate_paired"] = evaluate
    try:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = True
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        with torch.autocast("cpu", dtype=torch.bfloat16):
            if failure is None:
                run("paired-010", ns)
            else:
                with pytest.raises(RuntimeError if failure == "evaluation" else AssertionError):
                    run("paired-010", ns)
            assert torch.is_autocast_enabled("cpu")
            assert torch.get_autocast_dtype("cpu") == torch.bfloat16
            assert torch.get_float32_matmul_precision() == precision
            assert torch.backends.cuda.matmul.allow_tf32 == old_matmul_tf32
            assert torch.backends.cudnn.allow_tf32
        assert all(p.dtype == torch.float32 for p in ns["gpt"].layers[0].parameters())
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cudnn.allow_tf32 = old_cudnn


def test_model_load_default_is_explicit_fp32(namespace, monkeypatch):
    loader = Mock(return_value=namespace["gpt"])
    namespace["GPT2LensModel"] = SimpleNamespace(from_pretrained=loader)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    run("paired-005", namespace)
    loader.assert_called_once_with(
        "openai-community/gpt2-xl", device="cpu", dtype=torch.float32,
    )


def test_identity_rejects_reduced_weights_without_casting(namespace):
    ns = namespace
    ns["gpt"].layers[0].to(torch.bfloat16)
    ns["evals"] = {"order-ops": []}
    ns["evaluate_paired"] = Mock()
    old_precision = torch.get_float32_matmul_precision()
    old_cudnn = torch.backends.cudnn.allow_tf32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="requires fp32 weights"):
            run("paired-010", ns)
        assert torch.is_autocast_enabled("cpu")
    assert torch.get_float32_matmul_precision() == old_precision
    assert torch.backends.cudnn.allow_tf32 == old_cudnn
    assert all(p.dtype == torch.bfloat16 for p in ns["gpt"].layers[0].parameters())
    ns["evaluate_paired"].assert_not_called()


def test_fit_rejects_active_autocast_without_downloading(namespace):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="outside autocast"):
            run("paired-fit-run", namespace)
        assert torch.is_autocast_enabled("cpu")
    namespace["load_fit_prompts"].assert_not_called()
    namespace["fit_with_progress"].assert_not_called()
