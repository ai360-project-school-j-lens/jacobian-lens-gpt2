# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Lens directions and band interventions on the tiny CPU decoder."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from jlens.lens import JacobianLens
from jlens.metrics import spearman_lens_vs_logits
from jlens.workspace import (
    ClampSwap,
    InjectDirection,
    final_norm_centers,
    lens_direction,
    record_residuals,
    swap_basis,
)
from tests.tiny import TinyDecoder

PROMPT = "hello workspace"


def _model(norm: str = "layer") -> TinyDecoder:
    model = TinyDecoder(n_layers=4, d_model=8, vocab_size=32)
    if norm == "rms":
        model.norm = nn.RMSNorm(model.d_model)
        with torch.no_grad():
            model.norm.weight.normal_(mean=1.0, std=0.1)
    return model


def _lens(model: TinyDecoder, seed: int = 0) -> JacobianLens:
    generator = torch.Generator().manual_seed(seed)
    jacobians = {
        layer: torch.eye(model.d_model)
        + 0.1 * torch.randn(model.d_model, model.d_model, generator=generator)
        for layer in range(model.n_layers - 1)
    }
    return JacobianLens(jacobians=jacobians, n_prompts=1, d_model=model.d_model)


def test_final_norm_centers_distinguishes_layernorm_from_rmsnorm():
    assert final_norm_centers(_model("layer")) is True
    assert final_norm_centers(_model("rms")) is False


@pytest.mark.parametrize("norm", ["layer", "rms"])
def test_lens_direction_matches_the_readout_it_claims_to_carry(norm):
    """``v_t . h`` must track the lens logit of ``t`` as ``h`` moves along ``v_t``.

    The whole point of the direction: pushing the residual along it raises that
    token's lens logit. A wrong centering term breaks this on one norm or the other.
    """
    model = _model(norm)
    lens = _lens(model)
    layer, token = 1, 7
    centers = final_norm_centers(model)
    direction = lens_direction(model, lens, token, layer, mode="J", centers=centers)

    base = torch.randn(1, model.d_model)
    transported = lens.transport(base, layer)
    before = model.unembed(transported)[0, token]
    nudged = lens.transport(base + 0.05 * direction / direction.norm(), layer)
    after = model.unembed(nudged)[0, token]
    assert after > before


def test_lens_direction_centering_differs_between_norms():
    """The centering branch is load-bearing, not a no-op."""
    model = _model("layer")
    lens = _lens(model)
    centered = lens_direction(model, lens, 7, 1, mode="J", centers=True)
    uncentered = lens_direction(model, lens, 7, 1, mode="J", centers=False)
    assert not torch.allclose(centered, uncentered)


def test_lens_direction_logit_mode_ignores_the_jacobian():
    model = _model("rms")
    direction = lens_direction(model, None, 7, 1, mode="logit")
    gamma = model.norm.weight.float()
    assert torch.allclose(direction, gamma * model.lm_head.weight[7].float())


def test_lens_direction_rejects_bad_arguments():
    model = _model()
    with pytest.raises(ValueError, match="mode must be"):
        lens_direction(model, _lens(model), 1, 1, mode="tuned")
    with pytest.raises(ValueError, match="needs a fitted lens"):
        lens_direction(model, None, 1, 1, mode="J")


def test_swap_basis_pseudoinverse_round_trips():
    model = _model()
    V, V_pinv = swap_basis(model, _lens(model), 3, 9, 1)
    assert V.shape == (model.d_model, 2)
    assert V_pinv.shape == (2, model.d_model)
    assert torch.allclose(V_pinv @ V, torch.eye(2), atol=1e-4)


def test_swap_basis_coordinates_recover_a_vector_in_the_span():
    """``V^+`` must read back the coordinates of anything inside ``span(V)``."""
    model = _model()
    V, V_pinv = swap_basis(model, _lens(model), 3, 9, 1)
    coords = torch.tensor([0.7, -1.3])
    assert torch.allclose(V_pinv @ (V @ coords), coords, atol=1e-4)


@pytest.mark.parametrize("norm", ["layer", "rms"])
def test_clamp_swap_is_a_noop_at_alpha_zero(norm):
    model = _model(norm)
    lens = _lens(model)
    layers = [1, 2]
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, layers)
    plain = model.forward(ids).last_hidden_state.clone()
    with ClampSwap(model, lens, layers, 3, 9, alpha=0.0, clean=clean):
        patched = model.forward(ids).last_hidden_state
    assert torch.allclose(plain, patched, atol=1e-5)


def test_clamp_swap_exchanges_the_two_coordinates_at_alpha_one():
    model = _model()
    lens = _lens(model)
    layers = [1]
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, layers)
    V, V_pinv = swap_basis(model, lens, 3, 9, 1)
    before = clean[1][0].float() @ V_pinv.T
    with ClampSwap(model, lens, layers, 3, 9, alpha=1.0, clean=clean):
        after_run = record_residuals(model, ids, layers)
    after = after_run[1][0].float() @ V_pinv.T
    assert torch.allclose(after, before[:, [1, 0]], atol=1e-4)


def test_clamp_swap_preserves_the_orthogonal_component():
    """Only ``span(V)`` may move; the rest of the residual is the model's own."""
    model = _model()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    V, _ = swap_basis(model, lens, 3, 9, 1)
    projector = V @ torch.linalg.pinv(V)
    orthogonal = torch.eye(model.d_model) - projector
    with ClampSwap(model, lens, [1], 3, 9, alpha=1.0, clean=clean):
        patched = record_residuals(model, ids, [1])
    assert torch.allclose(
        clean[1][0].float() @ orthogonal.T,
        patched[1][0].float() @ orthogonal.T,
        atol=1e-4,
    )


def test_clamp_swap_restricts_itself_to_the_given_positions():
    model = _model()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    with ClampSwap(
        model, lens, [1], 3, 9, alpha=1.0, clean=clean, positions=slice(1, None)
    ):
        patched = record_residuals(model, ids, [1])
    assert torch.allclose(clean[1][0, 0], patched[1][0, 0], atol=1e-6)
    assert not torch.allclose(clean[1][0, 1:], patched[1][0, 1:], atol=1e-4)


def test_clamp_swap_random_mode_matches_the_update_norm():
    model = _model()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    sizes = {}
    for mode in ("J", "random"):
        with ClampSwap(
            model, lens, [1], 3, 9, alpha=1.0, mode=mode, clean=clean,
            generator=torch.Generator().manual_seed(0),
        ):
            patched = record_residuals(model, ids, [1])
        sizes[mode] = (patched[1][0] - clean[1][0]).float().norm(dim=-1)
    assert torch.allclose(sizes["J"], sizes["random"], rtol=1e-4)


def test_clamp_swap_removes_its_hooks_on_exit():
    model = _model()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    plain = model.forward(ids).last_hidden_state.clone()
    with ClampSwap(model, lens, [1], 3, 9, alpha=1.0, clean=clean):
        pass
    assert torch.allclose(model.forward(ids).last_hidden_state, plain, atol=1e-6)


def test_inject_direction_is_a_noop_at_strength_zero():
    model = _model()
    ids = model.encode(PROMPT)
    plain = model.forward(ids).last_hidden_state.clone()
    with InjectDirection(model, _lens(model), [1, 2], 7, strength=0.0):
        patched = model.forward(ids).last_hidden_state
    assert torch.allclose(plain, patched, atol=1e-6)


def test_inject_direction_scales_with_strength_and_residual_norm():
    """The update is ``strength * mean||h|| * unit(v)``, so it is linear in strength."""
    model = _model()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    deltas = {}
    for strength in (0.5, 1.0):
        with InjectDirection(model, lens, [1], 7, strength=strength):
            patched = record_residuals(model, ids, [1])
        deltas[strength] = (patched[1][0] - clean[1][0]).float()
    assert torch.allclose(deltas[1.0], 2 * deltas[0.5], atol=1e-5)

    expected = 0.5 * clean[1][0].float().norm(dim=-1).mean()
    assert torch.allclose(
        deltas[0.5].norm(dim=-1), expected.expand(deltas[0.5].shape[0]), rtol=1e-4
    )


def test_inject_direction_points_along_the_lens_direction():
    model = _model("rms")
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    with InjectDirection(model, lens, [1], 7, strength=1.0):
        patched = record_residuals(model, ids, [1])
    delta = (patched[1][0, 0] - clean[1][0, 0]).float()
    wanted = lens_direction(model, lens, 7, 1, mode="J", centers=False)
    cosine = torch.dot(delta, wanted) / (delta.norm() * wanted.norm())
    assert cosine > 0.999


def test_spearman_matches_scipy_including_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    generator = torch.Generator().manual_seed(0)
    lens_scores = torch.randn(200, generator=generator)
    model_logits = torch.randn(200, generator=generator)
    ids = [3, 7, 11, 42, 99, 100, 150, 199]
    assert spearman_lens_vs_logits(lens_scores, model_logits, ids) == pytest.approx(
        scipy_stats.spearmanr(lens_scores[ids], model_logits[ids]).statistic, abs=1e-5
    )

    tied_lens = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0])
    tied_model = torch.tensor([5.0, 4.0, 3.0, 2.0, 2.0])
    assert spearman_lens_vs_logits(
        tied_lens, tied_model, range(5)
    ) == pytest.approx(
        scipy_stats.spearmanr(tied_lens, tied_model).statistic, abs=1e-5
    )


def test_spearman_is_nan_when_undefined():
    scores = torch.randn(10, generator=torch.Generator().manual_seed(0))
    assert spearman_lens_vs_logits(scores, scores, [4]) != spearman_lens_vs_logits(
        scores, scores, [4]
    )  # NaN
    constant = torch.zeros(10)
    value = spearman_lens_vs_logits(constant, scores, [0, 1, 2])
    assert value != value


def test_spearman_ignores_duplicate_candidates():
    generator = torch.Generator().manual_seed(1)
    lens_scores = torch.randn(50, generator=generator)
    model_logits = torch.randn(50, generator=generator)
    ids = [1, 4, 9, 16]
    assert spearman_lens_vs_logits(
        lens_scores, model_logits, ids
    ) == pytest.approx(
        spearman_lens_vs_logits(lens_scores, model_logits, ids + ids), abs=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_random_mode_accepts_a_cpu_generator_on_a_cuda_model():
    """A ``torch.Generator`` is device-bound; seeding a CUDA draw from a CPU one must
    still work, so a run stays reproducible without knowing where the model lives."""
    model = _model().cuda()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    for intervention in (
        ClampSwap(
            model, lens, [1], 3, 9, alpha=1.0, mode="random", clean=clean,
            generator=torch.Generator().manual_seed(0),
        ),
        InjectDirection(
            model, lens, [1], 7, strength=1.0, mode="random",
            generator=torch.Generator().manual_seed(0),
        ),
    ):
        with intervention:
            model.forward(ids)


@pytest.mark.parametrize(
    "make",
    [
        lambda model, lens, clean: InjectDirection(
            model, lens, [1], 7, strength=1.0, positions=slice(2, 5)
        ),
        lambda model, lens, clean: ClampSwap(
            model, lens, [1], 3, 9, alpha=1.0, clean=clean, positions=slice(2, 5)
        ),
    ],
    ids=["inject", "swap"],
)
def test_band_hooks_are_a_noop_when_the_position_slice_selects_nothing(make):
    """A KV-cached decode step passes one new token, so a prompt-position slice is empty.

    Reducing over that empty slice writes NaN into the residual stream, which is silent:
    generation keeps running and produces garbage. The hook must pass the output through.
    """
    model = _model()
    lens = _lens(model)
    ids = model.encode(PROMPT)
    clean = record_residuals(model, ids, [1])
    single = ids[:, -1:]
    plain = model.forward(single).last_hidden_state.clone()
    with make(model, lens, clean):
        patched = model.forward(single).last_hidden_state
    assert torch.isfinite(patched).all()
    assert torch.allclose(plain, patched, atol=1e-6)
