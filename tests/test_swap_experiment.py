"""Offline regression checks for the model-agnostic token swap experiment."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.swap_experiment import (
    BatchedSwap,
    experiment_fingerprint,
    intermediate_bands,
    load_swap_trials,
    prepare_swap_trials,
    run_swap_experiment,
)
from jlens.workspace import final_norm_gain
from tests.tiny import TinyDecoder

ROOT = Path(__file__).resolve().parents[1]


def lens_for(model):
    return JacobianLens(
        {layer: torch.eye(model.d_model) for layer in range(model.n_layers)},
        n_prompts=1, d_model=model.d_model,
    )


def prepared_rows():
    return pd.DataFrame([
        dict(trial_id=str(i), dataset="synthetic", group="test", name=str(i),
             prompt=str(i), source="a", target="b", answer="c", swap_answer="d",
             input_ids=ids, source_id=source, target_id=target,
             answer_ids=[4, 5], swap_answer_ids=[6, 7], eligible=True)
        for i, (ids, source, target) in enumerate([
            ([0, 1, 2, 3], 3, 9), ([0, 2, 3], 4, 10), ([0, 5, 6, 7], 5, 11),
        ])
    ])


class PaddedTiny(TinyDecoder):
    supports_attention_mask = True

    def forward(self, input_ids, *, attention_mask=None):
        # A causal cross-token model makes wrong readout positions observable.
        hidden = self.embed_tokens(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        hidden = hidden.cumsum(dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.mark.parametrize("model_type", [TinyDecoder, PaddedTiny])
def test_batch_matches_individual_and_noop(model_type):
    model = model_type().eval()
    lens = lens_for(model)
    options = dict(bands={"middle": [1, 2]}, alphas=[0, 1], progress=False)
    rows = prepared_rows()
    together = run_swap_experiment(model, lens, rows, batch_size=3, **options)
    separate = run_swap_experiment(model, lens, rows, batch_size=1, **options)
    keys = ["trial_id", "band", "alpha", "mode"]
    # Random draws depend on batching; deterministic arms must agree.
    a = together[together['mode'] != 'random'].sort_values(keys)
    b = separate[separate['mode'] != 'random'].sort_values(keys)
    for column in ["top", "base_top", "rank", "success", "probability"]:
        torch.testing.assert_close(torch.tensor(a[column].tolist()), torch.tensor(b[column].tolist()))
    assert not together[together.alpha == 0].changed.any()
    assert all(not layer._forward_hooks for layer in model.layers)


def test_coordinate_exchange_positions_random_norms_and_cleanup():
    model = TinyDecoder().eval()
    ids = torch.tensor([[0, 1, 2], [0, 3, 4]])
    with ActivationRecorder(model.layers, at=[1]) as recorder:
        model.forward(ids)
    clean = {1: recorder.activations[1].detach()}
    basis = torch.randn(2, model.d_model, 2)
    positions = torch.tensor([[False, True, True], [False, True, False]])
    with BatchedSwap(model, {1: basis}, clean, positions) as swap:
        with ActivationRecorder(model.layers, at=[1]) as recorder:
            model.forward(ids)
    patched = recorder.activations[1]
    inverse = torch.linalg.pinv(basis)
    clean_coords = clean[1] @ inverse.transpose(1, 2)
    patched_coords = patched @ inverse.transpose(1, 2)
    torch.testing.assert_close(patched_coords[positions], clean_coords.flip(-1)[positions])
    torch.testing.assert_close(patched[~positions], clean[1][~positions])
    delta = patched - clean[1]
    projection = (delta @ inverse.transpose(1, 2)) @ basis.transpose(1, 2)
    torch.testing.assert_close(delta, projection, atol=1e-6, rtol=1e-5)
    with BatchedSwap(model, {1: basis}, clean, positions, random_norms=swap.update_norms) as random:
        model.forward(ids)
    torch.testing.assert_close(random.update_norms[1], swap.update_norms[1])
    with pytest.raises(RuntimeError), BatchedSwap(model, {1: basis}, clean, positions):
        raise RuntimeError("test cleanup")
    assert not model.layers[1]._forward_hooks


class OffsetRMS(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-0.4, 0.6, width))

    def forward(self, x):
        return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + self.weight)


def test_effective_gain_supports_offset_rms_and_layernorm_bias():
    model = TinyDecoder()
    model.norm = OffsetRMS(model.d_model)
    gain = final_norm_gain(model)
    expected = 1 + model.norm.weight
    torch.testing.assert_close(gain / gain.norm(), expected / expected.norm())
    model.norm = nn.LayerNorm(model.d_model)
    with torch.no_grad():
        model.norm.bias.uniform_(-2, 2)
        model.norm.weight.uniform_(0.1, 2)
    gain = final_norm_gain(model)
    expected = model.norm.weight
    torch.testing.assert_close(gain / gain.norm(), expected / expected.norm())


class Tokenizer:
    bos_token_id = 0
    all_special_ids = [0, 1]
    vocabulary = {"a": 2, " b": 3, "yes": 4, " no": 5, " ": 6}

    def encode(self, text, **kwargs):
        return [self.vocabulary[text]] if text in self.vocabulary else [6, 7]

    def decode(self, ids, **kwargs):
        return next((word for word, token in self.vocabulary.items() if ids == [token]), "?")


def test_token_audit_rejects_fragments_and_overlength_without_losing_bos():
    model = TinyDecoder()
    model.tokenizer = Tokenizer()
    model.encode_ids = lambda prompt, max_length: [0, 2, 3][:max_length]
    base = dict(dataset="test", name="test", prompt="prompt", source="a", target="b",
                answer="yes", swap_answer="no", quality_ok=True)
    trials = pd.DataFrame([base, dict(base, target="multiword"), dict(base, swap_answer="nonsense"),
                           dict(base, swap_answer="yes")])
    prepared = prepare_swap_trials(model, trials)
    assert prepared.eligible.tolist() == [True, False, False, False]
    assert prepared.iloc[0].input_ids == [0, 2, 3]
    assert prepared.iloc[0].source_id == 2  # Bare form when space-prefixed form splits.
    assert not prepare_swap_trials(model, trials, max_length=2).eligible.any()


def test_audit_provenance_counts_and_exclusions():
    import hashlib
    import json

    frame = load_swap_trials(ROOT)
    assert frame.groupby("dataset").size().to_dict() == {
        "probe-swap": 90, "latent-bridge": 400, "flexible-generalization": 192,
    }
    assert not frame.trial_id.duplicated().any()
    assert not frame[frame.name == "super-populous-capital"].quality_ok.any()
    assert not frame[frame.group == "numbers/square"].quality_ok.any()
    assert frame[frame.dataset == "latent-bridge"].prompt.nunique() == 100
    audit = json.loads((ROOT / "data/experiments/token-swap-audit.json").read_text())
    for name, expected in audit["upstream"]["files"].items():
        assert hashlib.sha256((ROOT / "data/experiments" / name).read_bytes()).hexdigest() == expected


def test_cache_identity_and_invalid_bands():
    model = TinyDecoder()
    lens = lens_for(model)
    rows = prepared_rows()
    original = experiment_fingerprint({"model": "tiny"}, rows, lens)
    assert original != experiment_fingerprint({"model": "other"}, rows, lens)
    lens.jacobians[1][0, 0] += 1
    assert original != experiment_fingerprint({"model": "tiny"}, rows, lens)
    with pytest.raises(ValueError, match="intermediate"):
        run_swap_experiment(model, lens, rows, bands={"last": [3]})
    assert all(max(band) < 3 for band in intermediate_bands(4).values())


def test_hf_final_readout_with_right_padding():
    from transformers import GPT2Config, GPT2LMHeadModel

    from jlens import from_hf
    from tests.tiny import _ByteTokenizer

    torch.manual_seed(12)
    hf_model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_embd=8, n_layer=4, n_head=2, n_positions=32,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )).eval()
    model = from_hf(hf_model, _ByteTokenizer())
    frame = prepared_rows()
    result = run_swap_experiment(model, lens_for(model), frame, bands={"middle": [1]},
                                 alphas=[0], batch_size=3, progress=False)
    ids = torch.tensor([[0, 1, 2, 3], [0, 2, 3, 0], [0, 5, 6, 7]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    with torch.no_grad():
        expected = hf_model(ids, attention_mask=mask).logits[torch.arange(3), torch.tensor([3, 2, 3])]
    actual = result.drop_duplicates("trial_id").sort_values("trial_id")
    assert actual.base_top.tolist() == expected.argmax(-1).tolist()
    expected_probability = expected.softmax(-1)[:, [6, 7]].sum(-1)
    torch.testing.assert_close(torch.tensor(actual.base_probability.tolist()), expected_probability)


def test_tuple_outputs_are_preserved_and_degenerate_pairs_rejected():
    model = TinyDecoder().eval()
    basis = torch.randn(1, model.d_model, 2)
    clean = {1: torch.randn(1, 3, model.d_model)}
    hook = BatchedSwap(model, {1: basis}, clean, torch.ones(1, 3, dtype=torch.bool))._hook(1)
    result = hook(None, None, (clean[1].clone(), "cache", 7))
    assert result[1:] == ("cache", 7)
    frame = prepared_rows()
    frame["target_id"] = frame.source_id
    with pytest.raises(ValueError, match="Degenerate"):
        run_swap_experiment(model, lens_for(model), frame, bands={"middle": [1]}, progress=False)
