"""Offline checks for Colab bootstrap and exact benchmark lens selection."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
from transformers import Gemma4Config, LlamaConfig

import jlens.artifact_provenance as provenance

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/model_agnostic/latent_bridge_v2_lens_benchmark.ipynb"
)


def cell_source(cell_id):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return next(cell.source for cell in notebook.cells if cell.id == cell_id)


@pytest.mark.parametrize("location", ["local", "nested_colab", "existing_colab", "fresh_colab"])
def test_repository_bootstrap(tmp_path, monkeypatch, location):
    repo = tmp_path / "jacobian-lens-gpt2"

    def checkout():
        (repo / "jlens").mkdir(parents=True, exist_ok=True)
        for name in ("evaluation.py", "position_benchmark.py"):
            (repo / "jlens" / name).touch()

    if location != "fresh_colab":
        checkout()
    working = repo / "notebooks" if location in {"local", "nested_colab"} else tmp_path
    working.mkdir(exist_ok=True)
    monkeypatch.chdir(working)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    if location != "local":
        monkeypatch.setenv("COLAB_RELEASE_TAG", "offline-test")

    def run(command, **kwargs):
        assert kwargs["check"] is True
        if command[:2] == ["git", "clone"]:
            checkout()

    runner = Mock(side_effect=run)
    monkeypatch.setattr(subprocess, "run", runner)
    state = {}
    exec(cell_source("benchmark-repository-setup"), state)
    assert state["REPO"] == repo.resolve()
    commands = [call.args[0] for call in runner.call_args_list]
    if location == "local":
        assert commands == []
    else:
        assert len(commands) == 2
        if location == "fresh_colab":
            assert commands[0][:2] == ["git", "clone"]
        else:
            assert commands[0] == ["git", "-C", str(repo), "pull", "--ff-only"]
        assert commands[1] == [sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)]


@pytest.mark.parametrize("selection", ["hub", "local", "smoke", "missing_preset", "download_failure"])
def test_lens_resolution_does_not_fall_back(tmp_path, monkeypatch, selection):
    local = tmp_path / "lens.pt"
    local.write_bytes(b"different local file")
    downloaded = tmp_path / "hub" / "lens.pt"
    receipt = {"path": str(downloaded), "sha256": "a" * 64, "resolved_revision": "b" * 40}
    download = Mock(return_value=SimpleNamespace(path=str(downloaded), to_dict=lambda: receipt))
    monkeypatch.setattr(provenance, "download_lens_artifact", download)
    request = dict(repo_id="offline/lens", filename="nested/lens.pt", revision="main")
    state = dict(
        Path=Path, display=Mock(), RUN_MODE="smoke" if selection == "smoke" else "pretrained",
        MODEL_ID="offline/model", FIT_PROVENANCE={"model_id": "offline/model"},
        LENS_SOURCE="local" if selection == "local" else "hub", LENS_PATH=local,
        HUB_LENS=None if selection == "missing_preset" else request,
        RESOLVED_LENS_PATH=local, LENS_PROVENANCE={"stale": True},
    )
    if selection == "download_failure":
        download.side_effect = OSError("download failed")
    source = cell_source("benchmark-lens-source")
    if selection in {"missing_preset", "download_failure"}:
        with pytest.raises((ValueError, OSError)):
            exec(source, state)
        assert state["RESOLVED_LENS_PATH"] is None
        assert state["LENS_PROVENANCE"] is None
    else:
        exec(source, state)
        expected = {"hub": downloaded, "local": local, "smoke": None}[selection]
        assert state["RESOLVED_LENS_PATH"] == expected
    if selection in {"hub", "download_failure"}:
        download.assert_called_once_with(**request)
    else:
        download.assert_not_called()


@pytest.mark.parametrize("revision", ["main", "release-tag", "c" * 40])
def test_gemma_config_without_private_commit_hash(monkeypatch, revision):
    """Resolve the Hub commit before loading any model-dependent artifacts."""
    config = Gemma4Config()
    config.__dict__.pop("_commit_hash", None)
    assert not hasattr(config, "_commit_hash")
    commit = "d" * 40
    api = Mock()
    api.model_info.return_value = SimpleNamespace(sha=commit)
    monkeypatch.setattr("huggingface_hub.HfApi", Mock(return_value=api))
    auto_config = Mock()
    auto_config.from_pretrained.return_value = config
    auto_tokenizer = Mock()
    auto_tokenizer.from_pretrained.return_value = SimpleNamespace(is_fast=True)
    state = dict(
        RUN_MODE="pretrained", MODEL_ID="google/gemma-4-31B", MODEL_REVISION=revision,
        FIT_PROVENANCE={"model_id": "google/gemma-4-31B", "model_revision": commit},
        AutoConfig=auto_config, AutoTokenizer=auto_tokenizer,
        LENS_SOURCE="hub", LENS_PROVENANCE={"sha256": "a" * 64},
        DEVICE="cpu", DTYPE="float32",
    )
    # Execute the actual identity/loading portion, before manifest serialization.
    source = cell_source("c1eb4c15").split("\nimplementation =", 1)[0]
    exec(source, state)
    assert state["resolved_revision"] == commit
    api.model_info.assert_called_once_with("google/gemma-4-31B", revision=revision)
    auto_config.from_pretrained.assert_called_once_with("google/gemma-4-31B", revision=commit)
    auto_tokenizer.from_pretrained.assert_called_once_with(
        "google/gemma-4-31B", revision=commit, use_fast=True,
    )


@pytest.mark.parametrize("config", [
    LlamaConfig(vocab_size=129), Gemma4Config(text_config={"vocab_size": 129}),
], ids=["text", "multimodal"])
def test_position_plot_uses_text_vocabulary(tmp_path, monkeypatch, config):
    """Run the actual plot cell with flat and nested HF configurations."""
    names = ["logit lens", "J-lens"]
    words = pd.DataFrame([
        {"dataset": "example", "item": "toy", "kind": "intermediate",
         "word": "country", "lens": name, "positions": [1, 2, 3],
         "ranks": np.array([[100, 50, 20], [30, 10, 5], [10, 5, 1]])}
        for name in names
    ])
    spans = pd.concat([
        words.assign(span=span, ranks=[row.min(1) for row in words.ranks])
        for span in ["subject", "country token", "all but slot"]
    ], ignore_index=True)
    state = dict(
        plt=plt, np=np, pd=pd, N_LAYERS=3, span_words=spans, position_words=words,
        latent_items=pd.DataFrame([{
            "dataset": "example", "item": "toy", "ids": [0, 1, 2, 3],
            "prompt": "Toy country prompt",
        }]),
        tokenizer=SimpleNamespace(decode=lambda ids: str(ids[0])),
        config=config, COLORS=dict(zip(names, ["red", "blue"], strict=True)),
        K=10, RUN_MODE="smoke", RUN_DIR=tmp_path,
    )
    monkeypatch.setattr(plt, "show", Mock())
    previous = set(plt.get_fignums())
    try:
        exec(cell_source("de8d805f"), state)
        image = state["fig"].axes[0].images[0]
        assert image.norm.vmax == pytest.approx(np.log10(129))
        assert (tmp_path / "position_example.png").is_file()
    finally:
        for number in set(plt.get_fignums()) - previous:
            plt.close(number)
