"""Exercise the notebook's pretrained setup with local tiny model artifacts."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest
import torch

import jlens
import jlens.vis as vis
from jlens.evaluation import identity_lens

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/model_agnostic/jacobian_lens/interactive_jacobian_lens.ipynb"


@pytest.fixture
def notebook_state(tmp_path, monkeypatch):
    for name in ("JLENS_EXPLORER_MODE", "JLENS_MODEL_ID", "JLENS_LENS_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(vis._TEMPLATE_FOR_MODE, "embed", vis._template("fetch"))
    cells = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    state = dict(REPO=REPO, Path=Path, os=os, json=json, hashlib=hashlib)
    for name in ("imports", "configuration", "session-helpers"):
        exec(cells[name], state)
    state["CACHE_ROOT"] = tmp_path / "cache"
    smoke = state["prepare_session"]()
    tokenizer = smoke.tokenizer
    config = state["AutoConfig"].for_model("llama", **{
        key: value for key, value in smoke.manifest["config"].items()
        if key != "model_type"
    })
    model = state["AutoModelForCausalLM"].from_config(config).eval()
    lens_path = tmp_path / "lens.pt"
    identity_lens(jlens.from_hf(model, tokenizer, compile=False)).save(
        str(lens_path), dtype=torch.float32,
    )
    smoke.close()
    commit = "a" * 40
    state.update(
        RUN_MODE="pretrained", MODEL_ID="offline/test-model", MODEL_REVISION="release",
        LENS_PATH=lens_path, DEVICE="cpu", DTYPE=torch.float32,
        FIT_PROVENANCE={"model_id": "offline/test-model", "model_revision": commit},
        HUB_LENS={"repo_id": "offline/lenses", "filename": "lens.pt", "revision": commit},
    )
    state["HfApi"] = Mock(return_value=SimpleNamespace(
        model_info=Mock(return_value=SimpleNamespace(sha=commit)),
    ))
    state["AutoConfig"] = Mock()
    state["AutoConfig"].from_pretrained.return_value = config
    state["AutoTokenizer"] = Mock()
    state["AutoTokenizer"].from_pretrained.return_value = tokenizer
    state["AutoModelForCausalLM"] = Mock()
    state["AutoModelForCausalLM"].from_pretrained.return_value = model
    receipt = dict(path=str(lens_path), sha256=state["digest_file"](lens_path),
                   resolved_revision=commit)
    state["download_lens_artifact"] = Mock(return_value=SimpleNamespace(
        path=str(lens_path), to_dict=lambda: receipt,
    ))
    return state, cells, commit


@pytest.mark.parametrize("source", ["hub", "local"])
def test_pretrained_loading_pins_all_artifacts_and_waits_for_a_prompt(notebook_state, source):
    state, _, commit = notebook_state
    state["LENS_SOURCE"] = source
    explorer = state["prepare_session"]()
    try:
        state["AutoConfig"].from_pretrained.assert_called_once_with(
            "offline/test-model", revision=commit,
        )
        state["AutoTokenizer"].from_pretrained.assert_called_once_with(
            "offline/test-model", revision=commit, use_fast=True,
        )
        if source == "hub":
            state["download_lens_artifact"].assert_called_once_with(**state["HUB_LENS"])
        else:
            state["download_lens_artifact"].assert_not_called()
        state["AutoModelForCausalLM"].from_pretrained.assert_not_called()
        result = explorer.render("1 + 2 = ")
        assert not result.cache_hit
        state["AutoModelForCausalLM"].from_pretrained.assert_called_once_with(
            "offline/test-model", revision=commit, dtype=torch.float32,
        )
        assert explorer.render("1 + 2 = ").cache_hit
        state["BOS_POLICY"] = "none"
        with pytest.raises(RuntimeError, match="Configuration changed"):
            explorer.render("1 + 2 = ")
    finally:
        explorer.close()


def test_failed_preparation_retires_old_session_and_never_falls_back(notebook_state):
    state, cells, _ = notebook_state
    state["LENS_SOURCE"] = "hub"
    previous = state["prepare_session"]()
    state["explorer"] = previous
    state["download_lens_artifact"].side_effect = OSError("offline download failure")
    with pytest.raises(OSError, match="download failure"):
        exec(cells["prepare-session"], state)
    assert "explorer" not in state
    with pytest.raises(RuntimeError, match="closed"):
        previous.render("ab")
    state["AutoModelForCausalLM"].from_pretrained.assert_not_called()
