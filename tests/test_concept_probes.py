"""CPU tests for concept probe fitting and causal swap integration."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from jlens.concept_probes import (
    audit_probe_templates,
    collect_probe_features,
    fit_concept_probes,
    prepare_probe_examples,
    probe_directions,
    probe_fingerprint,
)
from jlens.hooks import ActivationRecorder
from jlens.swap_experiment import (
    ConceptDirections,
    experiment_fingerprint,
    prepare_swap_trials,
    run_swap_experiment,
)
from tests.test_swap_experiment import PaddedTiny, lens_for, prepared_rows
from tests.test_swap_experiment import Tokenizer as AnswerTokenizer
from tests.tiny import TinyDecoder

TEMPLATES = ["Topic: {x}.", "Read about {x} today.", "Write a note about {x}.", "Subject: {x}!"]
CONCEPTS = ["red fox", "blue whale", "green bird"]


def probe_model(model_type=PaddedTiny):
    words = ["[UNK]", "[BOS]", "[PAD]", "Topic", ":", "Read", "about", "today", ".",
             "Write", "a", "note", "Subject", "!", "red", "fox", "blue", "whale", "green", "bird"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="[BOS]", pad_token="[PAD]")
    model = model_type().eval()
    model.tokenizer = tokenizer
    model.encode_ids = lambda text, max_length: tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
    return model


def examples_for(model):
    return prepare_probe_examples(model, CONCEPTS, audit_probe_templates(TEMPLATES, held_out=1))


def test_template_audit_rejects_suffix_leakage_and_malformed_slots():
    audit = audit_probe_templates(TEMPLATES, held_out=1)
    assert audit.split.value_counts().to_dict() == {"train": 3, "test": 1}
    pd.testing.assert_frame_equal(audit, audit_probe_templates(TEMPLATES, held_out=1))
    with pytest.raises(ValueError, match="causal prefix"):
        audit_probe_templates(["About {x}.", "About {x}!"], held_out=1)
    with pytest.raises(ValueError, match="slot"):
        audit_probe_templates(["About {x} and {x}", "Other {x}"], held_out=1)


@pytest.mark.parametrize("model_type", [PaddedTiny, TinyDecoder])
def test_multi_token_span_bos_and_batched_features(model_type):
    model = probe_model(model_type)
    examples = examples_for(model)
    assert examples.valid.all()
    assert examples.concept_tokens.eq(2).all()
    assert all(ids[0] == model.tokenizer.bos_token_id for ids in examples.input_ids)
    features = collect_probe_features(model, examples, [1, 2], batch_size=7, progress=False)
    for i, row in enumerate(examples.to_dict("records")):
        ids = torch.tensor([row["input_ids"]])
        with torch.no_grad(), ActivationRecorder(model.layers, at=[1, 2]) as recorder:
            model.forward(ids)
        for layer in [1, 2]:
            torch.testing.assert_close(features[layer][i], recorder.activations[layer][0, row["position"]])
    assert all(not block._forward_hooks for block in model.layers)


def test_bad_probe_spans_are_reported():
    model = probe_model()
    templates = audit_probe_templates(TEMPLATES, held_out=1)
    unknown = prepare_probe_examples(model, ["unknown concept"], templates)
    assert not unknown.valid.any()
    assert unknown.reason.str.contains("unknown").all()
    truncated = prepare_probe_examples(model, CONCEPTS, templates, max_length=3)
    assert not truncated.valid.any()
    model.encode_ids = lambda text, max_length: [2]
    mismatched = prepare_probe_examples(model, CONCEPTS, templates)
    assert mismatched.reason.eq("offset encoding differs from model encoding").all()


def test_fit_uses_training_only_and_solves_ridge_objective():
    model = probe_model()
    examples = examples_for(model)
    rng = torch.Generator().manual_seed(22)
    labels = torch.tensor([CONCEPTS.index(c) for c in examples.concept])
    features = {1: torch.randn(len(examples), 5, generator=rng) * 0.05 + torch.eye(3, 5)[labels]}
    result = fit_concept_probes(features, examples, ridge=0.1)
    assert all(row["test_accuracy"] == 1 for row in result["diagnostics"])
    tampered = {1: features[1].clone()}
    tampered[1][examples.split.eq("test").to_numpy().copy()] = 999
    other = fit_concept_probes(tampered, examples, ridge=0.1)
    for mode in result["directions"]:
        torch.testing.assert_close(result["directions"][mode][1], other["directions"][mode][1])
    # Independent normal-equation check, after returning weights to residual units.
    train = examples.split.eq("train").to_numpy().copy()
    x = features[1][train].double()
    y = torch.nn.functional.one_hot(torch.tensor([result["concepts"].index(c) for c in examples.loc[train, "concept"]]), 3).double()
    xc, yc = x - x.mean(0), y - y.mean(0)
    scale_squared = xc.square().mean()
    expected = torch.linalg.solve(xc.T @ xc + len(x) * 0.1 * scale_squared * torch.eye(5), xc.T @ yc)
    torch.testing.assert_close(result["directions"]["probe_ridge"][1].T.double(), expected, atol=1e-6, rtol=1e-5)


def test_probe_fitting_requires_split_and_sufficient_concepts():
    examples = examples_for(probe_model())
    features = {1: torch.randn(len(examples), 5)}
    bad = examples.copy()
    bad.loc[bad.split.eq("test"), "prefix"] = bad.loc[bad.split.eq("train"), "prefix"].iloc[0]
    with pytest.raises(ValueError, match="overlap"):
        fit_concept_probes(features, bad)
    with pytest.raises(ValueError, match="three concepts"):
        fit_concept_probes({1: features[1][:4]}, examples.iloc[:4])


def test_learned_arms_match_manual_clamp_and_allow_multi_token_cohort():
    from jlens.swap_experiment import BatchedSwap

    model = TinyDecoder().eval()
    frame = prepared_rows().iloc[[0, 2]].copy()
    frame["source"] = "red fox"
    frame["target"] = "blue whale"
    matrix = torch.randn(3, model.d_model)
    concepts = ("red fox", "blue whale", "green bird")
    supplied = {"probe_mean": ConceptDirections(concepts, {1: matrix})}
    frame[["source_id", "target_id"]] = None
    result = run_swap_experiment(model, lens_for(model), frame, bands={"middle": [1]},
                                 alphas=[0, 1], token_controls=False, concept_directions=supplied,
                                 batch_size=2, progress=False)
    assert set(result['mode']) == {"probe_mean", "random_probe_mean"}
    assert not result[result.alpha.eq(0)].changed.any()
    ids = torch.tensor(frame.input_ids.tolist())
    with torch.no_grad(), ActivationRecorder(model.layers, at=[1]) as recorder:
        model.forward(ids)
    basis = matrix[:2].T.unsqueeze(0).expand(2, -1, -1)
    positions = ids != 0
    with torch.no_grad(), BatchedSwap(model, {1: basis}, recorder.activations, positions):
        with ActivationRecorder(model.layers, at=[3]) as patched:
            model.forward(ids)
    expected = model.unembed(patched.activations[3][:, -1]).argmax(-1).tolist()
    assert result[(result['mode'] == 'probe_mean') & result.alpha.eq(1)].top.tolist() == expected
    fingerprint = experiment_fingerprint({}, frame, lens_for(model), concept_directions=supplied)
    supplied["probe_mean"] = replace(supplied["probe_mean"], vectors={1: matrix + 1})
    assert fingerprint != experiment_fingerprint({}, frame, lens_for(model), concept_directions=supplied)


def test_token_and_probe_cohorts_are_explicit_and_answers_stay_strict():
    model = TinyDecoder()
    model.tokenizer = AnswerTokenizer()
    model.encode_ids = lambda prompt, max_length: [0, 2, 3]
    trial = dict(prompt="prompt", source="red fox", target="blue whale", answer="yes",
                 swap_answer="no", quality_ok=True)
    frame = pd.DataFrame([trial, dict(trial, swap_answer="nonsense")])
    assert not prepare_swap_trials(model, frame).eligible.any()
    assert prepare_swap_trials(model, frame, require_single_token_concepts=False).eligible.tolist() == [True, False]


def test_feature_cache_depends_on_spans_split_and_config():
    examples = examples_for(probe_model())
    base = probe_fingerprint({"model": "tiny"}, examples, [1, 2])
    assert base != probe_fingerprint({"model": "other"}, examples, [1, 2])
    assert base != probe_fingerprint({"model": "tiny"}, examples, [1])
    changed = examples.copy()
    changed["position"] -= 1
    assert base != probe_fingerprint({"model": "tiny"}, changed, [1, 2])


def test_repository_probe_templates_are_valid_and_independent_of_tasks():
    import json

    from jlens.swap_experiment import load_swap_trials

    root = Path(__file__).resolve().parents[1]
    templates = json.loads((root / "data/experiments/concept-probe-templates.json").read_text())["templates"]
    audit = audit_probe_templates(templates)
    assert len(audit) == 20
    assert audit.prefix.nunique() == 20
    trials = load_swap_trials(root)
    prompts = {template.format(x=concept).strip() for template in templates
               for concept in set(trials.source) | set(trials.target)}
    assert not prompts.intersection(trials.prompt.str.strip())


def test_probe_directions_roundtrip_and_dual_ridge():
    model = probe_model()
    examples = examples_for(model)
    features = collect_probe_features(model, examples, [1], progress=False)
    features[1] = torch.cat([features[1], features[1]], dim=-1)  # n_train < d -> dual solve.
    result = fit_concept_probes(features, examples)
    sets = probe_directions(result)
    assert sets['probe_ridge'].vectors[1].shape == (3, 16)
    assert sets['probe_mean'].concepts == tuple(sorted(CONCEPTS))


def test_probe_only_runner_needs_only_lens_model_protocol():
    model = TinyDecoder().eval()

    class ProtocolOnly:
        n_layers = model.n_layers
        d_model = model.d_model
        layers = model.layers
        tokenizer = model.tokenizer
        encode = model.encode
        forward = model.forward
        unembed = model.unembed

    frame = prepared_rows()
    supplied = {"learned": ConceptDirections(("a", "b", "c"), {1: torch.randn(3, model.d_model)})}
    result = run_swap_experiment(ProtocolOnly(), lens_for(model), frame,
                                 bands={"middle": [1]}, alphas=[0, 1], token_controls=False,
                                 concept_directions=supplied, progress=False)
    assert len(result) == len(frame) * 4
    assert not result[result.alpha.eq(0)].changed.any()
