"""Offline artifact verification and both dataset notebooks' download/load wiring."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest
import torch

import jlens.artifact_provenance as provenance
from jlens.lens import JacobianLens
from jlens.notebook_setup import DatasetFitRun

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / name
    for name in (
        "model_agnostic/model_agnostic_lens_dataset.ipynb",
        "gemma-4-31b/jacobian_lens/failed_gemma_model_agnostic_lens_dataset.ipynb",
    )
]
COMMIT = "a" * 40
REQUEST = dict(repo_id="offline/lenses", filename="nested/lens.pt", revision="main")


def mock_hub(monkeypatch, path):
    data = path.read_bytes()
    entry = SimpleNamespace(
        rfilename=REQUEST["filename"],
        lfs=SimpleNamespace(size=len(data), sha256=hashlib.sha256(data).hexdigest()),
    )
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=COMMIT, siblings=[entry])
    monkeypatch.setattr(provenance, "HfApi", Mock(return_value=api))
    download = Mock(return_value=str(path))
    monkeypatch.setattr(provenance, "hf_hub_download", download)
    return api, download, entry


def test_exact_revision_and_published_digest_ignore_stale_basename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "lens.pt").write_bytes(b"stale")
    (tmp_path / "lens.pt.1").write_bytes(b"also stale")
    downloaded = tmp_path / "hub" / "lens.pt"
    downloaded.parent.mkdir()
    downloaded.write_bytes(b"intended artifact")
    api, download, _ = mock_hub(monkeypatch, downloaded)
    artifact = provenance.download_lens_artifact(**REQUEST)
    api.model_info.assert_called_once_with(
        REQUEST["repo_id"], revision="main", files_metadata=True,
    )
    download.assert_called_once_with(
        repo_id=REQUEST["repo_id"], filename=REQUEST["filename"], revision=COMMIT,
    )
    assert artifact.verify() == downloaded
    assert artifact.resolved_revision == COMMIT
    assert artifact.to_dict()["sha256"] == hashlib.sha256(downloaded.read_bytes()).hexdigest()
    assert artifact.to_dict()["fitting_provenance"] == "unverified legacy final lens"
    downloaded.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        artifact.verify()


@pytest.mark.parametrize("damage", [
    "missing_file", "missing_lfs", "bad_hash", "bad_size", "no_commit", "corrupt", "network",
])
def test_fail_closed(tmp_path, monkeypatch, damage):
    path = tmp_path / "lens.pt"
    path.write_bytes(b"downloaded bytes")
    api, download, entry = mock_hub(monkeypatch, path)
    if damage == "missing_file":
        api.model_info.return_value.siblings = []
    elif damage == "missing_lfs":
        entry.lfs = None
    elif damage == "bad_hash":
        entry.lfs.sha256 = "not a sha256"
    elif damage == "bad_size":
        entry.lfs.size = None
    elif damage == "no_commit":
        api.model_info.return_value.sha = "main"
    elif damage == "corrupt":
        # Same size detects that verification is not merely a length check.
        path.write_bytes(b"X" * path.stat().st_size)
    else:
        download.side_effect = OSError("network failed")
    with pytest.raises((ValueError, OSError)):
        provenance.download_lens_artifact(**REQUEST)
    if damage not in ("corrupt", "network"):
        download.assert_not_called()


def notebook_cells(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return {cell.id: cell.source for cell in notebook.cells}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("failure", [None, "corrupt", "network", "conflict"])
def test_notebook_loads_verified_download_not_local_stale_file(
    tmp_path, monkeypatch, notebook, failure,
):
    cells = notebook_cells(notebook)
    monkeypatch.chdir(tmp_path)
    intended = JacobianLens({0: torch.eye(2) * 3}, n_prompts=7, d_model=2)
    path = tmp_path / "hub-lens.pt"
    intended.save(str(path))
    (tmp_path / "gemma-4-31B_jacobian_lens.pt").write_bytes(b"stale")
    (tmp_path / "gemma-4-31B_jacobian_lens.pt.1").write_bytes(b"wrong duplicate")
    _, download, _ = mock_hub(monkeypatch, path)
    if failure == "corrupt":
        path.write_bytes(b"truncated")
    elif failure == "network":
        download.side_effect = OSError("network failed")
    model, hf_model, tokenizer = SimpleNamespace(d_model=2, n_layers=2), object(), object()
    model.tokenizer = tokenizer
    config = dict(model_id="offline/tiny", dtype="float32", max_seq_len=8, run_label="test")
    fit_run = DatasetFitRun(tmp_path, config)
    ns = dict(
        Path=Path, model=model, hf_model=hf_model, configured_objects=(model, hf_model, tokenizer),
        current_fit_config=lambda: config, FIT_CONFIG=config, fit_run=fit_run,
        EXISTING_LENS_PATH=path if failure == "conflict" else None, HUB_LENS=REQUEST.copy(),
        MODEL_ID="offline/tiny", DTYPE=torch.float32, DEVICE="cpu", REPO_DIR=tmp_path,
        RUN_DIR=fit_run.directory, ALLOW_NEW_FIT_WITH_LEGACY=False,
        runtime_metadata=lambda *args: {"model_id": "offline/tiny"},
        legacy_final_candidates=lambda *args: [], display=Mock(),
        load_fit_prompts=Mock(), fit_with_progress=Mock(), benchmark_dim_batches=Mock(),
        fitted_lens="stale notebook state", words="old rank cache", items="old metrics",
        reference52="old plot data", held_out_metrics="old held-out cache",
    )
    source = cells["generic-fit-run"]
    if failure:
        with pytest.raises((ValueError, OSError)):
            exec(compile(source, str(notebook), "exec"), ns)
        assert ns["fitted_lens"] is None
    else:
        exec(compile(source, str(notebook), "exec"), ns)
        assert ns["LENS_PATH"] == path
        assert ns["fitted_lens"].n_prompts == 7
        assert torch.equal(ns["fitted_lens"].jacobians[0], torch.eye(2) * 3)
        receipt = json.loads((fit_run.directory / "downloaded_lens_provenance.json").read_text())
        assert receipt["resolved_revision"] == COMMIT
        assert receipt["path"] == str(path)
        assert receipt["evaluation_runtime"] == {"model_id": "offline/tiny"}
    for key in ("words", "items", "reference52", "held_out_metrics"):
        assert ns[key] is None
    for key in ("load_fit_prompts", "fit_with_progress", "benchmark_dim_batches"):
        ns[key].assert_not_called()


def test_historical_outputs_explicitly_labeled_and_download_not_executable():
    cells = notebook_cells(NOTEBOOKS[1])
    assert "NOT RERUN" in cells["generic-000"]
    assert "every saved output" in cells["generic-000"]
    assert "Historical conclusions" in cells["generic-056"]
    assert "!wget" not in "\n".join(cells.values())
    assert "repo_id='neuronpedia/jacobian-lens'" in cells["generic-012"]
    assert "EXISTING_LENS_PATH = None" in cells["generic-012"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_both_notebooks_warn_old_results_must_be_recomputed(notebook):
    cells = notebook_cells(notebook)
    assert "NOT RERUN" in cells["generic-000"]
    assert "stale" in cells["generic-000"]
    assert "must be recomputed" in cells["generic-000"]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_handwritten_callback_preserves_both_readout_spaces(notebook):
    from jlens.readout import LensReadout

    ns = {"torch": torch, "LensModel": object}
    exec(compile(notebook_cells(notebook)["generic-008"], str(notebook), "exec"), ns)
    logits = torch.ones(1, 3)
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    model = SimpleNamespace(
        n_layers=2, unembed_readout=Mock(return_value=LensReadout(logits, scores)),
    )
    result = ns["handwritten_logit_lens"](model, {0: logits[None], 1: logits[None]})
    assert torch.equal(result.logits, torch.stack([logits, logits]))
    assert torch.equal(result.ranking_scores, torch.stack([scores, scores]))
