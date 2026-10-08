"""Offline distribution/geometry correctness; no downloaded models."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from pandas.testing import assert_frame_equal
from torch import nn

from jlens.lens import JacobianLens
from jlens.metrics import evaluate_distributions, lens_vector_geometry


class ToyModel(nn.Module):
    """Two blocks with a known non-symmetric transport and linear readout."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.n_layers = self.d_model = 2
        self.tokenizer = SimpleNamespace(all_special_ids=[0, 3], bos_token_id=0)
        self.embedding = nn.Embedding(4, 2, dtype=dtype)
        self.layers = nn.ModuleList(
            [nn.Identity(), nn.Linear(2, 2, bias=False, dtype=dtype)]
        )
        self.lm_head = nn.Linear(2, 4, bias=False, dtype=dtype)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.tensor([[0.0, 0.0], [2.0, -1.0], [-1.0, 1.0], [9.0, 9.0]])
            )
            self.layers[1].weight.copy_(torch.tensor([[1.0, 2.0], [0.0, -1.0]]))
            self.lm_head.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
            )
        self.forward_calls = 0
        self.grad_enabled = []
        self.eval()

    def encode(self, text, *, max_length=512):
        return torch.tensor(
            [[0] + [int(t) for t in text]], device=self.embedding.weight.device
        )[:, :max_length]

    def forward(self, ids):
        self.forward_calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        hidden = self.embedding(ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def unembed(self, residual):
        return self.lm_head(residual.to(self.lm_head.weight))


def make_lens(matrix):
    # Final-layer garbage is intentionally ignored by the shared-final convention.
    return JacobianLens(
        {0: matrix, 1: torch.full((2, 2), float("nan"))}, n_prompts=1, d_model=2
    )


def direct_reference(model, matrix, tokens):
    h = model.embedding(torch.tensor(tokens)).float()
    final = model.unembed(model.layers[1](h)).float().log_softmax(-1)
    baseline = model.unembed(h).float().log_softmax(-1)
    jacobian = model.unembed(h @ matrix.T).float().log_softmax(-1)
    return [baseline, final, jacobian, final]


def test_directions_pairwise_half_sum_and_token_aggregation():
    model = ToyModel()
    matrix = torch.tensor([[0.5, 1.0], [-0.25, 2.0]])
    result = evaluate_distributions(
        model,
        make_lens(matrix),
        ["1", "2223", ""],
        position_chunk_size=2,
        progress=False,
    )
    expected = direct_reference(model, matrix, [1, 2, 2, 2])
    assert model.forward_calls == 2  # special-only text is skipped
    assert model.grad_enabled == [False, False]
    assert len(result.layers) == 4
    assert len(result.pairs) == 2
    assert result.pairs.layer_a.tolist() == [0, 1]
    assert result.pairs.layer_b.tolist() == [0, 1]
    assert result.pairs.lens_a.tolist() == ["logit lens"] * 2
    assert result.pairs.lens_b.tolist() == ["J-lens"] * 2
    assert set(result.layers.n_tokens) == {4}
    assert set(result.layers.n_texts) == {3}
    assert set(result.layers.n_texts_used) == {2}
    assert set(result.layers.weighting) == {"token"}
    final = expected[-1]
    for row, logp in zip(result.layers.itertuples(), expected, strict=True):
        kl = (final.exp() * (final - logp)).sum(-1, dtype=torch.float64).mean().item()
        entropy = -(logp.exp() * logp).sum(-1, dtype=torch.float64).mean().item()
        agreement = (logp.argmax(-1) == final.argmax(-1)).float().mean().item()
        assert row.kl_model_to_lens == pytest.approx(kl, abs=1e-7)
        assert row.entropy == pytest.approx(entropy, abs=1e-7)
        assert row.top1_agreement == agreement
    keys = [(r.lens, r.layer) for r in result.layers.itertuples()]
    for row in result.pairs.itertuples():
        a = expected[keys.index((row.lens_a, row.layer_a))]
        b = expected[keys.index((row.lens_b, row.layer_b))]
        half_sum = 0.5 * (
            (a.exp() * (a - b)).sum(-1, dtype=torch.float64)
            + (b.exp() * (b - a)).sum(-1, dtype=torch.float64)
        )
        assert row.symmetric_kl == pytest.approx(half_sum.mean().item(), abs=1e-7)
        assert (
            row.top1_agreement == (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        )
    # Distinguish KL direction and token weighting from plausible wrong implementations.
    reverse = (expected[0].exp() * (expected[0] - final)).sum(-1).mean().item()
    assert abs(result.layers.iloc[0].kl_model_to_lens - reverse) > 0.01
    token_kl = (final.exp() * (final - expected[0])).sum(-1)
    assert (
        abs(result.layers.iloc[0].kl_model_to_lens - token_kl[:2].mean().item()) > 0.01
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identity_shared_final_and_chunk_invariance(dtype):
    model = ToyModel(dtype)
    lens = make_lens(torch.eye(2))
    one = evaluate_distributions(
        model, lens, ["1212"], position_chunk_size=1, progress=False
    )
    many = evaluate_distributions(
        model, lens, ["1212"], position_chunk_size=8, progress=False
    )
    torch.testing.assert_close(
        torch.tensor(one.layers.select_dtypes("number").values),
        torch.tensor(many.layers.select_dtypes("number").values),
    )
    torch.testing.assert_close(
        torch.tensor(one.pairs.select_dtypes("number").values),
        torch.tensor(many.pairs.select_dtypes("number").values),
    )
    cross = one.pairs.query(
        "lens_a == 'logit lens' and lens_b == 'J-lens' and layer_a == layer_b"
    )
    assert (cross.symmetric_kl == 0).all()
    assert (cross.top1_agreement == 1).all()
    final = one.layers.query("is_final")
    assert (final.kl_model_to_lens == 0).all()
    assert (final.top1_agreement == 1).all()
    assert final.entropy.nunique() == 1


def test_true_transport_matches_model_not_transpose():
    model = ToyModel()
    lens = make_lens(model.layers[1].weight.detach().clone())
    result = evaluate_distributions(model, lens, ["12"], pairwise=False, progress=False)
    row = result.layers.query("lens == 'J-lens' and layer == 0").iloc[0]
    assert row.kl_model_to_lens == 0
    assert row.top1_agreement == 1
    assert result.pairs.empty
    assert "symmetric_kl" in result.pairs


def test_nonfinite_missing_empty_and_hook_cleanup():
    model = ToyModel()
    lens = make_lens(torch.eye(2))
    with pytest.raises(ValueError, match="no non-special"):
        evaluate_distributions(model, lens, ["", "33"], progress=False)
    with pytest.raises(ValueError, match="missing inner"):
        evaluate_distributions(
            model, JacobianLens({}, n_prompts=0, d_model=2), ["1"], progress=False
        )
    with torch.no_grad():
        model.lm_head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_distributions(model, lens, ["1"], progress=False)
    assert all(not layer._forward_hooks for layer in model.layers)
    with pytest.raises(ValueError, match="nonfinite"):
        lens_vector_geometry(model, lens, progress=False)


def test_geometry_orientation_chunking_final_and_zeros():
    model = ToyModel()
    matrix = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    lens = make_lens(matrix)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    table = lens_vector_geometry(
        model, lens, unembedding_weight=weight, vocab_chunk_size=1, progress=False
    )
    expected = (
        torch.nn.functional.cosine_similarity(weight[:3], weight[:3] @ matrix)
        .mean()
        .item()
    )
    assert table.iloc[0].mean_cosine == pytest.approx(expected, abs=1e-7)
    for chunk_size in (2, 3, 100):
        chunked = lens_vector_geometry(
            model,
            lens,
            unembedding_weight=weight,
            vocab_chunk_size=chunk_size,
            progress=False,
        )
        assert_frame_equal(table, chunked, atol=1e-7, rtol=1e-6)
    wrong = (
        torch.nn.functional.cosine_similarity(weight[:3], weight[:3] @ matrix.T)
        .mean()
        .item()
    )
    assert abs(expected - wrong) > 0.01
    assert table.iloc[1].mean_cosine == 1
    assert set(table.n_vocab) == {4}
    assert set(table.n_valid_vectors) == {3}
    assert set(table.n_zero_vectors) == {1}
    assert set(table.status) == {"ok"}
    identity = lens_vector_geometry(model, make_lens(torch.eye(2)), progress=False)
    assert identity.mean_cosine.tolist() == pytest.approx([1.0, 1.0])
    zero = lens_vector_geometry(model, make_lens(torch.zeros(2, 2)), progress=False)
    assert zero.iloc[0].status == "undefined"
    assert zero.iloc[0].n_valid_vectors == 0


def test_unsupported_geometry_does_not_linearize_nonlinear_decoder():
    model = ToyModel()
    model.lm_head = nn.Sequential(model.lm_head, nn.Tanh())
    result = lens_vector_geometry(model, make_lens(torch.eye(2)), progress=False)
    assert set(result.status) == {"unsupported"}
    assert result.mean_cosine.isna().all()
    assert result.reason.str.contains("linear head").all()


def test_normalized_decoder_uses_full_unembed():
    from jlens.hooks import ActivationRecorder

    from .tiny import TinyDecoder

    model = TinyDecoder(n_layers=2, d_model=4).eval()
    matrix = torch.diag(torch.tensor([2.0, 0.5, -1.0, 1.0]))
    lens = JacobianLens({0: matrix}, n_prompts=1, d_model=4)
    result = evaluate_distributions(model, lens, ["ab"], progress=False)
    with torch.no_grad(), ActivationRecorder(model.layers, at=[0, 1]) as recorder:
        model.forward(model.encode("ab"))
        inner = recorder.activations[0][0, 1:]
        final = model.unembed(recorder.activations[1][0, 1:]).log_softmax(-1)
        decoded = model.unembed(inner @ matrix.T).log_softmax(-1)
        expected = (final.exp() * (final - decoded)).sum(-1).mean().item()
        raw = model.lm_head(inner @ matrix.T).log_softmax(-1)
        wrong = (final.exp() * (final - raw)).sum(-1).mean().item()
    measured = result.layers.query("lens == 'J-lens' and layer == 0").iloc[0]
    assert measured.kl_model_to_lens == pytest.approx(expected, abs=1e-7)
    assert abs(expected - wrong) > 1e-3


def test_autocast_disabled_and_final_only_protocol_model():
    model = ToyModel()
    lens = JacobianLens({}, n_prompts=0, d_model=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = evaluate_distributions(model, lens, ["12"], layers=[], progress=False)
    assert result.layers.layer.tolist() == [1, 1]
    assert result.pairs.symmetric_kl.tolist() == [0.0]
    assert result.pairs.top1_agreement.tolist() == [1.0]


def test_many_layers_stream_readouts_and_linear_pairs(monkeypatch):
    import jlens.metrics as metrics

    model = ToyModel()
    model.layers.extend(nn.Identity() for _ in range(46))
    model.n_layers = 48
    lens = JacobianLens({i: torch.eye(2) for i in range(47)}, n_prompts=1, d_model=2)
    live_readouts = []
    decode = metrics._decode
    calls = 0

    def tracked_decode(*args, **kwargs):
        nonlocal calls
        result = decode(*args, **kwargs)
        calls += 1
        live_readouts[:] = [ref for ref in live_readouts if ref() is not None]
        live_readouts.append(weakref.ref(result[0]))
        # Shared final plus only this layer's baseline/transported distribution.
        assert len(live_readouts) <= 3
        return result

    monkeypatch.setattr(metrics, "_decode", tracked_decode)
    result = evaluate_distributions(
        model, lens, ["1212"], position_chunk_size=2, progress=False
    )
    assert calls == 2 * (2 * 48 - 1)
    assert len(result.layers) == 96
    assert len(result.pairs) == 48
    assert (result.pairs.layer_a == result.pairs.layer_b).all()
    assert all(ref() is None for ref in live_readouts)
    disabled = evaluate_distributions(
        model, lens, ["1212"], pairwise=False, progress=False
    )
    torch.testing.assert_close(
        torch.tensor(result.layers.select_dtypes("number").values),
        torch.tensor(disabled.layers.select_dtypes("number").values),
    )
    assert disabled.pairs.empty
    assert disabled.pairs.columns.tolist() == result.pairs.columns.tolist()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dtype_device_and_ambient_autocast(dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = ToyModel(dtype).to(device)
    lens = make_lens(torch.tensor([[0.5, 1.0], [-0.25, 2.0]]))  # CPU matrices
    original = lens.jacobians[0].clone()
    observed = []
    hook = model.layers[1].register_forward_hook(
        lambda _module, _args, output: observed.append(output.dtype)
    )
    try:
        reference = evaluate_distributions(model, lens, ["12"], progress=False)
        with torch.autocast(device, dtype=torch.bfloat16):
            result = evaluate_distributions(model, lens, ["12"], progress=False)
    finally:
        hook.remove()
    assert observed == [dtype, dtype]
    assert model.lm_head.weight.dtype == dtype
    assert lens.jacobians[0].device.type == "cpu"
    torch.testing.assert_close(lens.jacobians[0], original)
    assert reference.layers.equals(result.layers)
    assert reference.pairs.equals(result.pairs)
    final = result.pairs.iloc[-1]
    assert final.symmetric_kl == 0
    assert final.top1_agreement == 1


def test_finite_logits_with_nonfinite_log_probabilities_raise():
    model = ToyModel()
    maximum = torch.finfo(torch.float32).max
    model.unembed = lambda residual: torch.tensor(
        [maximum, -maximum], device=residual.device
    ).expand(len(residual), -1)
    with pytest.raises(ValueError, match="log probabilities: nonfinite"):
        evaluate_distributions(model, make_lens(torch.eye(2)), ["1"], progress=False)


def test_nonfinite_transport_raises_before_decoder_can_mask_it():
    model = ToyModel()
    model.unembed = lambda residual: torch.zeros(len(residual), 4)
    lens = make_lens(torch.full((2, 2), torch.finfo(torch.float32).max))
    with pytest.raises(ValueError, match="residual: nonfinite"):
        evaluate_distributions(model, lens, ["1"], progress=False)


@pytest.mark.parametrize("baseline_winner", [0, 1])
@pytest.mark.parametrize("jacobian_winner", [0, 1])
@pytest.mark.parametrize("final_winner", [0, 1])
def test_top1_uses_raw_logits_despite_normalization_ties(
    baseline_winner, jacobian_winner, final_winner
):
    model = ToyModel()
    lens = make_lens(-torch.eye(2))

    def near_tied_logits(residual):
        # Token 1 has first coordinate 2 at block 0, -2 after transport,
        # and 0 at the final block. All normalized argmaxes collapse to 0.
        winner = torch.where(
            residual[:, 0] == 2,
            baseline_winner,
            torch.where(residual[:, 0] == -2, jacobian_winner, final_winner),
        )
        logits = torch.stack([torch.zeros_like(winner), (2 * winner - 1) * 1e-8], -1)
        assert torch.equal(logits.argmax(-1), winner)
        assert (logits.log_softmax(-1).argmax(-1) == 0).all()
        return logits

    model.unembed = near_tied_logits
    result = evaluate_distributions(model, lens, ["11"], progress=False)
    assert result.layers.top1_agreement.tolist() == [
        float(baseline_winner == final_winner),
        1.0,
        float(jacobian_winner == final_winner),
        1.0,
    ]
    assert result.pairs.top1_agreement.tolist() == [
        float(baseline_winner == jacobian_winner),
        1.0,
    ]


@pytest.mark.parametrize(
    "layers", [[None], ["0"], [True], [False], [0.0], [[]], [-1], [2]]
)
@pytest.mark.parametrize("geometry", [False, True])
def test_invalid_layer_elements_raise_value_error(layers, geometry):
    model = ToyModel()
    lens = make_lens(torch.eye(2))
    with pytest.raises(ValueError, match="layers must be integer block indices"):
        if geometry:
            lens_vector_geometry(model, lens, layers=layers, progress=False)
        else:
            evaluate_distributions(model, lens, ["1"], layers=layers, progress=False)
    assert model.forward_calls == 0


@pytest.mark.parametrize("stage", ["input", "activation", "readout"])
def test_distribution_rejects_unsupported_devices_before_reductions(stage):
    # Meta tensors exercise guards without requiring MPS hardware.
    model = ToyModel()
    if stage == "input":
        model.encode = lambda *args, **kwargs: torch.empty(1, 2, device="meta")
    elif stage == "activation":
        model.layers[-1].register_forward_hook(
            lambda _module, _args, output: torch.empty_like(output, device="meta")
        )
    else:
        model.unembed = lambda residual: torch.empty(len(residual), 4, device="meta")
    with pytest.raises(ValueError, match="only CPU/CUDA compute devices; got meta"):
        evaluate_distributions(model, make_lens(torch.eye(2)), ["1"], progress=False)


@pytest.mark.parametrize("device", ["mps", "meta"])
def test_geometry_rejects_unsupported_compute_device(device):
    with pytest.raises(ValueError, match="only CPU/CUDA compute devices"):
        lens_vector_geometry(
            ToyModel(), make_lens(torch.eye(2)), device=device, progress=False
        )


def test_noncontiguous_subset_matches_full_four_layer_metrics():
    from .tiny import TinyDecoder

    model = TinyDecoder(n_layers=4, d_model=4).eval()
    lens = JacobianLens(
        {i: torch.eye(4) * (i + 0.5) for i in range(3)}, n_prompts=1, d_model=4
    )
    full = evaluate_distributions(model, lens, ["abc", "de"], progress=False)
    # Deliberately unordered/duplicated, omitting block 1 and the final block.
    subset_lens = JacobianLens(
        {i: lens.jacobians[i] for i in (0, 2)}, n_prompts=1, d_model=4
    )
    subset = evaluate_distributions(
        model, subset_lens, ["abc", "de"], layers=[2, 0, 2], progress=False
    )
    assert subset.layers.layer.tolist() == [0, 2, 3, 0, 2, 3]
    assert subset.pairs.layer_a.tolist() == [0, 2, 3]
    assert_frame_equal(
        subset.layers, full.layers.query("layer != 1").reset_index(drop=True)
    )
    assert_frame_equal(
        subset.pairs, full.pairs.query("layer_a != 1").reset_index(drop=True)
    )
    full_geometry = lens_vector_geometry(model, lens, progress=False)
    subset_geometry = lens_vector_geometry(
        model, subset_lens, layers=[2, 0, 2], progress=False
    )
    assert_frame_equal(
        subset_geometry, full_geometry.query("layer != 1").reset_index(drop=True)
    )
