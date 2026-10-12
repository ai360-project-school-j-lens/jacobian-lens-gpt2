"""Offline regression checks for the model-agnostic injection experiment."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from jlens import JacobianLens, from_hf
from jlens.introspection import (
    final_token_readout,
    introspection_bands,
    introspection_concepts,
    load_introspection_cache,
    prepare_introspection_prompt,
    run_introspection,
)
from jlens.readout import LensReadout
from jlens.workspace import InjectDirection, InjectDirections

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "data/experiments/verbal-introspection.json").read_text())


def tokenizer_fixture():
    splitter = pre_tokenizers.Whitespace()
    texts = [m["content"] for m in SPEC["intro_prompt"]]
    texts += [*SPEC["prefills"].values(), "User: Assistant: System:"]
    pieces = {p for text in texts for p, _ in splitter.pre_tokenize_str(text)}
    pieces.update(c["surface"] for c in SPEC["concepts"])
    vocab = {word: i for i, word in enumerate(["[UNK]", "[BOS]", "[EOS]", *sorted(pieces)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = splitter
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
    )


def model_fixture(architecture="llama"):
    tokenizer = tokenizer_fixture()
    torch.manual_seed(0)
    if architecture == "gpt2":
        config = GPT2Config(vocab_size=len(tokenizer), n_layer=3, n_embd=16, n_head=2, n_positions=256)
        hf = GPT2LMHeadModel(config).eval()
    else:
        config = LlamaConfig(
            vocab_size=len(tokenizer), num_hidden_layers=3, hidden_size=16,
            intermediate_size=32, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=256,
        )
        hf = LlamaForCausalLM(config).eval()
    model = from_hf(hf, tokenizer, force_bos=False)
    lens = JacobianLens(
        jacobians={l: torch.eye(16) for l in range(2)}, n_prompts=0, d_model=16,
    )
    prompt = prepare_introspection_prompt(tokenizer, SPEC)
    concepts = introspection_concepts(tokenizer, SPEC, prompt).iloc[:3].copy()
    return hf, model, lens, prompt, concepts


def test_prompt_plain_chat_bos_and_offsets():
    tokenizer = tokenizer_fixture()
    plain = prepare_introspection_prompt(tokenizer, SPEC)
    assert plain["style"] == "plain"
    assert plain["input_ids"].count(tokenizer.bos_token_id) == 1
    assert 0 not in plain["spans"]["whole prompt"]
    assert len(plain["input_ids"]) - 1 not in plain["spans"]["user turn"]
    question = tokenizer.decode([plain["input_ids"][p] for p in plain["spans"]["user turn"]])
    assert question.startswith("Trial 1") and question.endswith("about ?")
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    chat = prepare_introspection_prompt(tokenizer, SPEC, bos_policy="prepend")
    assert chat["style"] == "chat"
    assert chat["input_ids"].count(tokenizer.bos_token_id) == 1
    assert chat["spans"]["whole prompt"][-1] == len(chat["input_ids"]) - 1
    assert chat["spans"]["user turn"][-1] < len(chat["input_ids"]) - 1
    with pytest.raises(ValueError, match="no truncation"):
        prepare_introspection_prompt(tokenizer, SPEC, max_length=8)


def test_concept_coverage_never_uses_first_subtoken_or_unknown():
    tokenizer = tokenizer_fixture()
    spec = {**SPEC, "concepts": [
        {"name": "supported", "surface": "lightning"},
        {"name": "split", "surface": "lightning corner"},
        {"name": "unknown", "surface": "no_such_vocabulary_word"},
        {"name": "spaced", "surface": " lightning"},
    ]}
    table = introspection_concepts(tokenizer, spec, prepare_introspection_prompt(tokenizer, spec))
    assert table.supported.tolist() == [True, False, False, False]
    assert table.reason.tolist()[1:] == [
        "not one continuation token", "special or unknown token", "token does not decode to exact surface",
    ]
    assert table.token_id.isna().sum() == 3


def test_context_merging_is_excluded():
    tokenizer = tokenizer_fixture()
    # A tokenizer can have an exact concept token but merge it with the prefill.
    tokenizer.add_tokens(['"lightning'])
    prompt = prepare_introspection_prompt(tokenizer, SPEC)
    table = introspection_concepts(tokenizer, SPEC, prompt)
    assert table.iloc[0].reason == "prefill boundary retokenized"


@pytest.mark.parametrize("n_layers", [2, 4, 36, 48])
def test_fractional_bands_preserve_primary_and_exclude_final(n_layers):
    bands = introspection_bands(n_layers, {"wide middle": (1 / 3, 5 / 6), "late": (5 / 6, 47 / 48)})
    assert all(layers and min(layers) >= 0 and max(layers) < n_layers - 1 for layers in bands.values())
    if n_layers == 48:
        assert bands["wide middle"] == list(range(16, 40))


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_final_readout_matches_native_hf(architecture):
    hf, model, _, prompt, _ = model_fixture(architecture)
    ids = torch.tensor([prompt["input_ids"]]).expand(2, -1)
    with torch.no_grad():
        expected = hf(input_ids=ids).logits[:, -1]
    actual = final_token_readout(model, ids)
    torch.testing.assert_close(actual.logits, expected)


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_batched_sweep_matches_single_hook_and_generations(architecture):
    _, model, lens, prompt, concepts = model_fixture(architecture)
    output = run_introspection(
        model, lens, prompt, concepts, {"primary": [0, 1]}, [0.5, 2.0],
        primary_band="primary", batch_size=5, demo_concepts=1, demo_tokens=2,
    )
    trials = pd.DataFrame(output["trials"])
    paired = trials.pivot(index=["span", "band", "strength", "concept"], columns="mode", values="rank")
    np.testing.assert_array_equal(paired.J, paired.logit)
    control = trials[trials.strength.eq(0)]
    assert control.groupby("concept")["rank"].nunique().eq(1).all()
    for mode in ["J", "logit"]:
        with InjectDirection(model, lens, [0, 1], int(concepts.token_id.iloc[0]),
                             mode=mode, strength=0.5, positions=prompt["spans"]["whole prompt"]):
            logits = final_token_readout(model, torch.tensor([prompt["input_ids"]])).logits[0]
        target = int(concepts.token_id.iloc[0])
        ids = torch.arange(logits.numel())
        rank = 1 + int(((logits > logits[target]) | ((logits == logits[target]) & (ids < target))).sum())
        match = trials[trials.span.eq("whole prompt") & trials["mode"].eq(mode)
                       & trials.strength.eq(0.5) & trials.concept.eq(concepts.concept.iloc[0])]
        assert match["rank"].iloc[0] == rank
    for demo in output["demos"]:
        match = trials[trials.span.eq(demo["span"]) & trials["mode"].eq(demo["mode"])
                       & trials.strength.eq(demo["strength"]) & trials.concept.eq(demo["concept"])]
        assert demo["generated_ids"][0] == match.top1_id.iloc[0]
    assert all(not block._forward_hooks for block in model.layers)


def test_sweep_is_batch_size_independent_and_uses_one_clean_pass():
    _, model, lens, prompt, concepts = model_fixture()
    options = dict(primary_band="primary", demo_tokens=0, seed=17)
    large = run_introspection(model, lens, prompt, concepts, {"primary": [0]}, [1.0], batch_size=32, **options)
    small = run_introspection(model, lens, prompt, concepts, {"primary": [0]}, [1.0], batch_size=2, **options)
    assert large["trials"] == small["trials"]
    assert large["diagnostics"]["forward_passes"] == 1 + 2
    assert small["diagnostics"]["forward_passes"] == 1 + 2 * 5


def test_saturated_logits_preserve_greedy_rank_and_lexical_diagnostic(monkeypatch):
    _, model, lens, prompt, concepts = model_fixture()
    vocab_size = len(model.tokenizer)

    def saturated(residual):
        scores = (40 + torch.arange(vocab_size).float()).expand(len(residual), -1)
        return LensReadout(scores.tanh(), scores)

    monkeypatch.setattr(model, "unembed_readout", saturated)
    output = run_introspection(model, lens, prompt, concepts, {"primary": [0]}, [1.0],
                               primary_band="primary", demo_tokens=0)
    results = pd.DataFrame(output["trials"])
    assert results["rank"].eq(results.token_id + 1).all()
    assert results.lexical_rank.eq(vocab_size - results.token_id).all()
    assert results.top1_id.eq(0).all()


def test_continuations_patch_only_original_positions(monkeypatch):
    _, model, lens, prompt, concepts = model_fixture()
    original = InjectDirections._delta
    lengths = []

    def checked_delta(self, layer, hidden):
        lengths.append(hidden.shape[1])
        assert max(self._positions) < len(prompt["input_ids"])
        return original(self, layer, hidden)

    monkeypatch.setattr(InjectDirections, "_delta", checked_delta)
    run_introspection(model, lens, prompt, concepts, {"primary": [0]}, [1.0],
                      primary_band="primary", demo_tokens=3, demo_concepts=1)
    assert max(lengths) == len(prompt["input_ids"]) + 2


def cache_fixture(tmp_path):
    _, model, lens, prompt, concepts = model_fixture()
    output = run_introspection(model, lens, prompt, concepts, {"primary": [0]}, [1.0],
                               primary_band="primary", demo_tokens=0)
    output["manifest"] = dict(
        schema="introspection-v1", prompt=prompt, bands={"primary": [0]}, strengths=[1.0],
        selected_concepts=concepts.concept.tolist(), config={"id2label": {0: "label"}},
    )
    path = tmp_path / "results.json"
    path.write_text(json.dumps(output))
    return path, output


def test_cache_validates_pairs_and_source_manifest(tmp_path):
    path, payload = cache_fixture(tmp_path)
    before = path.read_bytes()
    assert load_introspection_cache(path, expected_manifest=payload["manifest"]) == json.loads(json.dumps(payload))
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_introspection_cache(path, expected_manifest={})
    payload["trials"].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete paired trial grid"):
        load_introspection_cache(path)


def test_missing_lens_layer_fails_before_inference():
    _, model, lens, prompt, concepts = model_fixture()
    del lens.jacobians[1]
    with pytest.raises(ValueError, match="missing an injection layer"):
        run_introspection(model, lens, prompt, concepts, {"primary": [1]}, [1.0], primary_band="primary")
    assert all(not block._forward_hooks for block in model.layers)


def test_notebook_cache_reanalysis_is_offline_and_preserves_source(tmp_path, monkeypatch):
    path = ROOT / "notebooks/model_agnostic/verbal_introspection/model_agnostic_verbal_introspection.ipynb"
    notebook = nbformat.read(path, as_version=4)
    source = {cell.id: cell.source for cell in notebook.cells}
    monkeypatch.setenv("JLENS_INTROSPECTION_MODE", "smoke")
    monkeypatch.setenv("JLENS_INTROSPECTION_ROOT", str(tmp_path))
    monkeypatch.setenv("JLENS_BOOTSTRAP_RESAMPLES", "32")
    state = dict(REPO=ROOT, os=os, Path=Path, json=json, hashlib=__import__("hashlib"))
    exec(source["imports"], state)
    state["display"] = Mock()
    for cell in ["configuration", "bootstrap-options", "cache-helpers", "identity", "run-or-restore"]:
        exec(source[cell], state)
    cache = state["CACHE_PATH"]
    before = cache.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Cache reanalysis must not download, load, or evaluate models")

    for target in [
        "transformers.AutoConfig.from_pretrained", "transformers.AutoTokenizer.from_pretrained",
        "transformers.AutoModelForCausalLM.from_config", "transformers.AutoModelForCausalLM.from_pretrained",
        "huggingface_hub.HfApi.model_info", "jlens.artifact_provenance.download_lens_artifact",
        "jlens.JacobianLens.load", "jlens.from_hf", "jlens.introspection.run_introspection",
        "jlens.introspection.final_token_readout",
    ]:
        monkeypatch.setattr(target, forbidden)
    state["run_introspection"] = forbidden
    # An automatic smoke cache hit must also skip weight creation and inference.
    for cell in ["identity", "run-or-restore"]:
        exec(source[cell], state)
    assert state["CACHE_PATH"] == cache

    # Cache mode ignores current model, dataset, lens and primary-setting defaults.
    monkeypatch.setenv("JLENS_INTROSPECTION_MODE", "cache")
    monkeypatch.setenv("JLENS_INTROSPECTION_CACHE", str(cache))
    monkeypatch.setenv("JLENS_MODEL_ID", "nonexistent/current-default")
    monkeypatch.setenv("JLENS_INTROSPECTION_DATASET", str(tmp_path / "missing.json"))
    monkeypatch.setenv("JLENS_LENS_PATH", str(tmp_path / "missing.pt"))
    exec(source["configuration"], state)
    state.update(PRIMARY_BAND="not the saved primary", PRIMARY_SPAN="not the saved span")
    for cell in ["identity", "prompt-audit", "run-or-restore", "controls", "point-estimates",
                 "analysis-destination", "bootstrap-analysis"]:
        exec(source[cell], state)
    assert state["manifest"]["model"] == "offline/random-tiny-llama"
    assert state["primary"].band.eq("wide middle").all()
    first_dir = state["ANALYSIS_DIR"]
    first_points = state["ci_table"].estimate.copy()
    state["BOOTSTRAP_OPTIONS"]["n_resamples"] = 64
    for cell in ["analysis-destination", "bootstrap-analysis"]:
        exec(source[cell], state)
    assert state["ANALYSIS_DIR"] != first_dir
    pd.testing.assert_series_equal(first_points, state["ci_table"].estimate)
    assert cache.read_bytes() == before
