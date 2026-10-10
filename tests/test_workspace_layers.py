"""Offline checks against direct vocabulary-space measurements."""

import numpy as np
import pytest
import torch

from jlens.evaluation import identity_lens
from jlens.lens import JacobianLens
from jlens.workspace_layers import (
    plot_workspace_layers,
    top1_autocorrelation,
    workspace_geometry,
    workspace_readouts,
)
from tests.tiny import TinyDecoder


def test_geometry_matches_explicit_centered_vocabulary_grams():
    model = TinyDecoder(n_layers=3, d_model=5)
    generator = torch.Generator().manual_seed(4)
    lens = JacobianLens(
        {
            0: torch.randn(5, 5, generator=generator),
            1: torch.randn(5, 5, generator=generator),
        },
        n_prompts=1,
        d_model=5,
    )
    weight = model.lm_head.weight.detach().double()
    result = workspace_geometry(
        model, lens, weight, layers=[0, 1], vocab_chunk_size=3, progress=False
    )
    grams = []
    for layer in result.layers:
        matrix = lens.jacobians[layer].double() if layer < 2 else torch.eye(5).double()
        vectors = weight @ matrix
        centered = vectors - vectors.mean(0)
        gram = centered @ centered.T
        grams.append((gram / gram.norm()).flatten())
        spectrum = torch.linalg.svdvals(centered).square()
        cumulative = spectrum.cumsum(0) / spectrum.sum()
        for row in result.dimensions.query("layer == @layer").itertuples():
            expected = (torch.searchsorted(cumulative, row.variance) + 1) / 5
            assert row.dimension_fraction == pytest.approx(float(expected))
    expected = torch.stack(grams) @ torch.stack(grams).T
    np.testing.assert_allclose(result.cka, expected.numpy(), atol=2e-6)
    # Vocabulary translation disappears under centering.
    shifted = workspace_geometry(
        model, lens, weight + 13, layers=[0, 1], progress=False
    )
    np.testing.assert_allclose(result.cka, shifted.cka, atol=2e-6)


def test_zero_geometry_is_undefined():
    model = TinyDecoder()
    result = workspace_geometry(
        model, identity_lens(model), torch.ones(32, 8), layers=[0], progress=False
    )
    assert np.isnan(result.cka).all()
    assert result.dimensions.dimension_fraction.isna().all()


def test_autocorrelation_exact_null_preserves_gaps_and_text_boundaries():
    # Text 1 has P(equal under shuffle)=1/3. Text 2 has P=1.
    result = top1_autocorrelation(
        [np.array([1, 1, -1, 2]), np.array([3, 3])], [1, 2, 10]
    ).set_index("lag")
    assert result.loc[1, "n_pairs"] == 2
    assert result.loc[1, "matches"] == 2
    assert result.loc[1, "null_expected_matches"] == pytest.approx(4 / 3)
    assert result.loc[1, "delta_log_p"] == pytest.approx(np.log(2.5 / (4 / 3 + 0.5)))
    assert result.loc[2, "n_pairs"] == 1  # do not compress the -1 gap
    assert result.loc[2, "matches"] == 0
    assert np.isnan(result.loc[10, "delta_log_p"])
    constant = top1_autocorrelation([np.array([7, 7, 7])], [1])
    assert constant.delta_log_p.iloc[0] == 0


def test_batched_readouts_match_direct_metrics_and_final_identity():
    model = TinyDecoder()
    lens = identity_lens(model)
    texts = ["cat", "a longer text", "fish", "", "x"]
    calls = []
    handle = model.layers[0].register_forward_hook(
        lambda _m, _i, output: calls.append(output.shape)
    )
    result = workspace_readouts(
        model,
        lens,
        texts,
        layers=[0, 2],
        ks=[1, 4],
        batch_size=3,
        position_chunk_size=2,
        progress=False,
    )
    handle.remove()
    assert len(calls) == 2  # sorted and padded, not one forward per text
    assert max(shape[0] for shape in calls) == 3
    assert set(result["accuracy"].n_tokens) == {sum(len(t) for t in texts)}
    assert result["accuracy"].query("layer == 3").accuracy.eq(1).all()
    direct_kurtosis = []
    with torch.no_grad():
        for text in texts:
            ids = model.encode(text)
            logits = model.unembed(model.forward(ids).last_hidden_state)[0, 1:]
            centered = logits - logits.mean(-1, keepdim=True)
            direct_kurtosis.extend(
                (
                    centered.pow(4).mean(-1) / centered.square().mean(-1).square() - 3
                ).tolist()
            )
    final_kurtosis = result["kurtosis"].query("layer == 3")
    np.testing.assert_allclose(
        final_kurtosis.excess_kurtosis,
        np.percentile(direct_kurtosis, final_kurtosis.percentile.to_numpy()),
        atol=2e-6,
    )
    unbatched = workspace_readouts(
        model,
        lens,
        texts,
        layers=[0, 2],
        ks=[1, 4],
        batch_size=1,
        position_chunk_size=99,
        progress=False,
    )
    for key in result:
        np.testing.assert_allclose(
            result[key], unbatched[key], atol=2e-6, equal_nan=True
        )


def test_plot_and_empty_position_handling():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = TinyDecoder()
    lens = identity_lens(model)
    with pytest.raises(ValueError, match="nonspecial"):
        workspace_readouts(model, lens, [""], layers=[0], progress=False)
    readouts = workspace_readouts(
        model, lens, ["test text"], layers=[0], progress=False
    )
    geometry = workspace_geometry(
        model, lens, model.lm_head.weight, layers=[0], progress=False
    )
    figures = plot_workspace_layers(geometry, readouts, n_layers=4, title="CPU test")
    assert len(figures[0].axes) == 2
    assert len(figures[1].axes) == 4
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)


def test_nonidentity_transport_accuracy_matches_direct_readout():
    model = TinyDecoder()
    lens = JacobianLens({0: torch.randn(8, 8)}, n_prompts=1, d_model=8)
    text = "nonidentity transport"
    result = workspace_readouts(
        model, lens, [text], layers=[0], ks=[1, 4, 32], progress=False
    )
    with torch.no_grad():
        h = model.layers[0](model.embed_tokens(model.encode(text)))[:, 1:]
        logits = model.unembed(h.float() @ lens.jacobians[0].T)[0]
        final = h
        for block in model.layers[1:]:
            final = block(final)
        target = model.unembed(final)[0].argmax(-1)
        for row in result["accuracy"].query("layer == 0").itertuples():
            expected = (
                (logits.topk(row.k).indices == target[:, None]).any(-1).float().mean()
            )
            assert row.accuracy == pytest.approx(float(expected))


def test_tiny_hf_causal_model_padding_and_final_readout():
    from transformers import GPT2Config, GPT2LMHeadModel

    from jlens import ActivationRecorder, from_hf

    torch.manual_seed(7)
    hf = GPT2LMHeadModel(
        GPT2Config(n_layer=2, n_head=2, n_embd=8, n_positions=64, vocab_size=32)
    ).eval()
    model = from_hf(hf, TinyDecoder().tokenizer, compile=False)
    lens = identity_lens(model)
    texts = ["short", "a much longer prompt", "last"]
    with torch.no_grad():
        ids = model.encode(texts[0])
        with ActivationRecorder(model.layers, at=[1]) as recorder:
            model.forward(ids)
        torch.testing.assert_close(
            model.unembed(recorder.activations[1]), hf(ids).logits
        )
    batched = workspace_readouts(
        model,
        lens,
        texts,
        layers=[0],
        batch_size=3,
        position_chunk_size=3,
        progress=False,
    )
    single = workspace_readouts(
        model,
        lens,
        texts,
        layers=[0],
        batch_size=1,
        position_chunk_size=50,
        progress=False,
    )
    for key in batched:
        np.testing.assert_allclose(batched[key], single[key], atol=1e-5, equal_nan=True)
