"""Offline bootstrap and prefitted-loading checks for the workspace notebook."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest
import torch
from huggingface_hub import HfApi

import jlens
from jlens.evaluation import identity_lens
from tests.tiny import TinyDecoder

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/model_agnostic/workspace_layers.ipynb"
)


@pytest.mark.parametrize("source", ["hub", "local"])
@pytest.mark.parametrize("failure", [None, "download", "width", "coverage", "nonfinite"])
def test_prefitted_notebook_never_fits(tmp_path, monkeypatch, source, failure):
    if source == "local" and failure == "download":
        pytest.skip("Local loading makes no download request")
    model = TinyDecoder(n_layers=3)
    lens = identity_lens(model)
    if failure == "width":
        lens = identity_lens(TinyDecoder(n_layers=3, d_model=4))
    elif failure == "coverage":
        lens.jacobians.pop(0)
    elif failure == "nonfinite":
        lens.jacobians[0][0, 0] = torch.nan
    path = tmp_path / "matching-final.pt"
    lens.save(str(path))
    download = Mock(return_value=str(path))
    if failure == "download":
        download.side_effect = OSError("Network unavailable")
    forbidden = Mock(side_effect=AssertionError("Prefitted loading must not enter fitting"))
    metadata = Mock(side_effect=AssertionError("Lens loading must not require LFS metadata"))
    monkeypatch.setattr(HfApi, "model_info", metadata)
    namespace = dict(
        Path=Path, torch=torch, jlens=jlens, model=model,
        RUN_MODE="hf", LENS_SOURCE=source,
        EXISTING_LENS_PATH=path,
        HUB_LENS=dict(repo_id="offline/lenses", filename="matching-final.pt", revision="main"),
        hf_hub_download=download, display=Mock(),
        DatasetFitRun=forbidden, fit_with_progress=forbidden,
        fit_prompt_loader=forbidden, benchmark_dim_batches=forbidden,
        fitted_lens="stale lens", readouts="stale readouts", geometry="stale geometry",
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(c for c in notebook.cells if c.id == "workspace-load-lens")
    if failure:
        with pytest.raises(OSError if failure == "download" else ValueError):
            exec(cell.source, namespace)
        assert namespace["fitted_lens"] is None
    else:
        exec(cell.source, namespace)
        assert namespace["fitted_lens"].d_model == model.d_model
        assert namespace["lens_path"] == path
        assert namespace["lens_provenance"]["source"] == source
        if source == "hub":
            assert namespace["lens_provenance"]["repo_id"] == "offline/lenses"
            assert namespace["lens_provenance"]["filename"] == "matching-final.pt"
    assert namespace["readouts"] is None
    assert namespace["geometry"] is None
    assert namespace["fit_run"] is None
    forbidden.assert_not_called()
    metadata.assert_not_called()
    if source == "local":
        download.assert_not_called()
    else:
        download.assert_called_once_with(**namespace["HUB_LENS"])


def _setup_source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return next(c.source for c in notebook.cells if c.id == "workspace-colab-setup")


def _checkout(path, *, current=True):
    (path / "jlens").mkdir(parents=True)
    (path / "jlens/evaluation.py").touch()
    if current:
        (path / "jlens/workspace_layers.py").touch()
        (path / "jlens/workspace_plotting.py").touch()


@pytest.mark.parametrize("colab_marker", ["environment", "module"])
def test_fresh_colab_clones_then_installs_in_active_interpreter(
    tmp_path, monkeypatch, colab_marker,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    if colab_marker == "environment":
        monkeypatch.setenv("COLAB_RELEASE_TAG", "offline-test")
    else:
        monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace())
    repository = tmp_path / "jacobian-lens-gpt2"

    def run(command, *, check):
        assert check is True
        if command[:2] == ["git", "clone"]:
            _checkout(Path(command[-1]))
        else:
            assert (repository / "jlens/workspace_layers.py").exists()
        return subprocess.CompletedProcess(command, 0)

    runner = Mock(side_effect=run)
    monkeypatch.setattr(subprocess, "run", runner)
    namespace = {}
    exec(_setup_source(), namespace)
    assert namespace["REPO_DIR"] == repository
    assert sys.path[0] == str(repository)
    assert [call.args[0] for call in runner.call_args_list] == [
        ["git", "clone", namespace["REPO_URL"], str(repository)],
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(repository)],
    ]


@pytest.mark.parametrize("location", ["inside", "alongside"])
def test_local_bootstrap_reuses_checkout_without_network(
    tmp_path, monkeypatch, location,
):
    repository = tmp_path / "jacobian-lens-gpt2"
    _checkout(repository)
    nested = repository / "notebooks/model_agnostic"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested if location == "inside" else tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    runner = Mock(side_effect=AssertionError("Unexpected network/install"))
    monkeypatch.setattr(subprocess, "run", runner)
    namespace = {}
    exec(_setup_source(), namespace)
    assert namespace["REPO_DIR"] == repository
    runner.assert_not_called()


def test_stale_checkout_stops_before_install(tmp_path, monkeypatch):
    _checkout(tmp_path / "jacobian-lens-gpt2", current=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "offline-test")
    runner = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    with pytest.raises(RuntimeError, match="pull --ff-only"):
        exec(_setup_source(), {})
    runner.assert_not_called()


def test_setup_detects_matplotlib_upgrade_before_model_loading(tmp_path, monkeypatch):
    _checkout(tmp_path / "jacobian-lens-gpt2")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "offline-test")
    monkeypatch.setitem(sys.modules, "matplotlib", SimpleNamespace(__version__="3.10.9"))
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "3.11.2")
    runner = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    with pytest.raises(RuntimeError, match="Restart the Colab"):
        exec(_setup_source(), {})
    runner.assert_called_once()  # Install completes, then the stale kernel stops.


@pytest.mark.parametrize(
    ("cuda", "bf16", "expected"),
    [(False, False, "float32"), (True, False, "float16"), (True, True, "bfloat16")],
)
def test_configuration_selects_supported_colab_dtype(tmp_path, cuda, bf16, expected):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    config = next(c.source for c in notebook.cells if c.cell_type == "code"
                  and c.source.startswith("RUN_MODE ="))
    native_bf16 = Mock(return_value=bf16)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=native_bf16),
        float32="float32", float16="float16", bfloat16="bfloat16",
    )
    namespace = dict(torch=fake_torch, Path=Path, REPO_DIR=tmp_path,
                     os=SimpleNamespace(environ={}))
    exec(config, namespace)
    assert namespace["RUN_MODE"] == "hf"
    assert namespace["DTYPE"] == expected
    assert namespace["MODEL_ID"] == "Qwen/Qwen3.5-9B"
    assert namespace["HUB_LENS"] == dict(
        repo_id="bcywinski/jacobian-lens-qwen3.5-9b",
        filename="lens_n1000.pt", revision="main",
    )
    if cuda:
        native_bf16.assert_called_once_with(including_emulation=False)
    else:
        native_bf16.assert_not_called()
