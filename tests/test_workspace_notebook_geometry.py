"""Exercise the notebook's own geometry and normalization code offline."""

import linecache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

from jlens.evaluation import identity_lens
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder

matplotlib.use("Agg")

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/model_agnostic/workspace_layers.ipynb"
)


@pytest.fixture
def notebook_code():
    import matplotlib.pyplot as plt

    namespace = dict(torch=torch, np=np, pd=pd, plt=plt, Path=Path)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.id.startswith("workspace-inline-") and cell.cell_type == "code":
            filename = f"<{cell.id}>"
            lines = cell.source.splitlines(keepends=True)
            linecache.cache[filename] = (len(cell.source), None, lines, filename)
            exec(compile(cell.source, filename, "exec"), namespace)
    return namespace


def test_gain_geometry_matches_explicit_vocabulary_grams(notebook_code):
    model = TinyDecoder(n_layers=3, d_model=5)
    generator = torch.Generator().manual_seed(42)
    lens = JacobianLens(
        {i: torch.randn(5, 5, generator=generator) for i in range(2)},
        n_prompts=1, d_model=5,
    )
    weight = model.lm_head.weight.detach().double()
    gain = torch.tensor([0.02, -0.5, 1.0, 3.0, 10.0], dtype=torch.float64)
    measure = notebook_code["workspace_geometry"]
    result = measure(model, lens, weight, layers=[0, 1], norm_gain=gain,
                     vocab_chunk_size=3, progress=False)
    grams = []
    for layer in result.layers:
        jacobian = lens.jacobians[layer].double() if layer < 2 else torch.eye(5).double()
        vectors = weight @ torch.diag(gain) @ jacobian
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
    raw = measure(model, lens, weight, layers=[0, 1], progress=False)
    scalar = measure(model, lens, weight, layers=[0, 1], norm_gain=torch.full((5,), 3.),
                     progress=False)
    np.testing.assert_allclose(raw.cka, scalar.cka, atol=2e-6)
    assert not np.allclose(raw.cka, result.cka, atol=0.01)
    zero = measure(model, lens, weight, layers=[0], norm_gain=torch.zeros(5),
                   progress=False)
    assert np.isnan(zero.cka).all()


@pytest.mark.parametrize("kind", ["gpt2", "qwen35", "gemma2", "gemma4"])
def test_real_norm_implementations_return_effective_gain(notebook_code, kind):
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    norm = {
        "gpt2": torch.nn.LayerNorm(8),
        "qwen35": Qwen3_5RMSNorm(8),
        "gemma2": Gemma2RMSNorm(8),
        "gemma4": Gemma4RMSNorm(8),
    }[kind]
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.75, 2, 8))
        if kind == "gpt2":
            norm.bias.fill_(4)  # Bias must not enter the gain.
    norm = norm.to(dtype=torch.bfloat16)
    before = norm.weight.detach().clone()
    gain, info = notebook_code["final_normalization_gain"](norm, 8)
    offset = int(kind in {"qwen35", "gemma2"})
    torch.testing.assert_close(gain, before.double() + offset)
    assert info["convention"] == ("1 + weight" if offset else "weight")
    torch.testing.assert_close(norm.weight, before)  # Probe never mutates model.


def test_gain_override_and_affine_free_norm(notebook_code):
    get_gain = notebook_code["final_normalization_gain"]
    gain, _ = get_gain(torch.nn.LayerNorm(5, elementwise_affine=False), 5)
    torch.testing.assert_close(gain, torch.ones(5).double())
    with pytest.raises(TypeError, match="NORM_GAIN_OVERRIDE"):
        get_gain(torch.nn.Identity(), 5)
    gain, info = get_gain(torch.nn.Identity(), 5, torch.arange(5))
    torch.testing.assert_close(gain, torch.arange(5).double())
    assert info["convention"] == "explicit override"
    with pytest.raises(ValueError, match="finite"):
        get_gain(torch.nn.Identity(), 5, [float("nan")] * 5)


@pytest.mark.parametrize("family", ["gpt2", "gemma2", "gemma4"])
def test_notebook_hf_loading_and_batched_readouts(notebook_code, monkeypatch, family):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Gemma2Config,
        Gemma4TextConfig,
        GPT2Config,
    )

    import jlens
    from jlens.notebook_setup import runtime_metadata
    from jlens.workspace_layers import workspace_readouts

    common = dict(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128,
    )
    if family == "gpt2":
        config = GPT2Config(vocab_size=32, n_positions=128, n_embd=16, n_layer=2, n_head=2)
    elif family == "gemma2":
        config = Gemma2Config(**common, query_pre_attn_scalar=8, sliding_window=64)
    else:
        config = Gemma4TextConfig(
            **common, global_head_dim=8, hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=32, layer_types=["sliding_attention", "full_attention"],
            sliding_window=64,
        )
    config._name_or_path = f"offline/{family}"
    hf = AutoModelForCausalLM.from_config(config)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", Mock(return_value=hf))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=TinyDecoder().tokenizer))
    namespace = notebook_code
    namespace.update(
        RUN_MODE="hf", MODEL_ID=f"offline/{family}", MODEL_REVISION=None,
        DTYPE=torch.float32, DEVICE="cpu", jlens=jlens, N_PLOT_LAYERS=25,
        runtime_metadata=runtime_metadata, load_fit_prompts=Mock(), display=Mock(),
    )
    cell = next(c for c in nbformat.read(NOTEBOOK, as_version=4).cells
                if c.id == "eaa87322")
    exec(cell.source, namespace)
    expected = hf.transformer.ln_f if family == "gpt2" else hf.model.norm
    assert namespace["final_norm"] is expected
    model = namespace["model"]
    gain, _ = namespace["final_normalization_gain"](expected, model.d_model)
    lens = identity_lens(model)
    for norm_gain in (None, gain):
        result = namespace["workspace_geometry"](
            model, lens, namespace["unembedding_weight"], layers=[0],
            norm_gain=norm_gain, progress=False,
        )
        np.testing.assert_allclose(result.cka, 1, atol=2e-6)
    batched = workspace_readouts(
        model, lens, ["short", "a longer prompt"], layers=[0], batch_size=2, progress=False,
    )
    individual = workspace_readouts(
        model, lens, ["short", "a longer prompt"], layers=[0], batch_size=1, progress=False,
    )
    for key in batched:
        np.testing.assert_allclose(batched[key], individual[key], atol=1e-5, equal_nan=True)


def test_geometry_cache_reuses_results_and_tracks_gain(notebook_code, tmp_path):
    import hashlib
    import json

    from jlens.notebook_setup import save_json

    namespace = notebook_code
    model = TinyDecoder(n_layers=3)
    namespace.update(
        hashlib=hashlib, json=json, model=model, fitted_lens=identity_lens(model),
        unembedding_weight=model.lm_head.weight, final_norm=model.norm,
        NORM_GAIN_OVERRIDE=None, runtime={"model_id": "tiny"},
        measurement_config={"lens_sha256": "offline"}, layers=[0, 1, 2],
        VARIANCE_THRESHOLDS=(0.9, 0.99), GEOMETRY_DEVICE="cpu", VOCAB_CHUNK_SIZE=3,
        CACHE_ROOT=tmp_path, save_json=save_json,
    )
    cell = next(c for c in nbformat.read(NOTEBOOK, as_version=4).cells
                if c.id == "workspace-geometry-cache")
    exec(cell.source, namespace)
    first_directory = namespace["GEOMETRY_DIR"]
    # Keep getsource() valid while making any cache-miss computation fail.
    monkey_model = SimpleNamespace(d_model=8, n_layers=3)
    namespace["model"] = monkey_model
    namespace["unembedding_weight"] = Mock(side_effect=AssertionError("Recomputed"))
    exec(cell.source, namespace)
    assert namespace["GEOMETRY_DIR"] == first_directory
    namespace["model"] = model
    namespace["unembedding_weight"] = model.lm_head.weight
    namespace["NORM_GAIN_OVERRIDE"] = torch.arange(1, 9)
    exec(cell.source, namespace)
    assert namespace["GEOMETRY_DIR"] != first_directory


def test_inline_plot_and_svg_fallback(notebook_code, tmp_path, monkeypatch):
    from matplotlib.backends.backend_svg import RendererSVG

    model = TinyDecoder()
    geometry = notebook_code["workspace_geometry"](
        model, identity_lens(model), model.lm_head.weight, layers=[0], progress=False,
    )
    figure = notebook_code["plot_cka"](
        geometry, n_layers=model.n_layers, title="Offline check",
        projection=r"$W_U \mathrm{diag}(\gamma) J_\ell$",
    )
    try:
        paths = notebook_code["save_workspace_figure"](figure, tmp_path / "gain")
        assert paths["png"].read_bytes().startswith(b"\x89PNG")
        assert "<svg" in paths["svg"].read_text()

        def broken_svg(*args, **kwargs):
            raise AttributeError("'Text' object has no attribute 'get_fontfeatures'")

        monkeypatch.setattr(RendererSVG, "_draw_text_as_path", broken_svg)
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            with pytest.warns(RuntimeWarning, match="PNG saved; SVG skipped"):
                paths = notebook_code["save_workspace_figure"](figure, tmp_path / "fallback")
        assert paths["png"].exists()
        assert paths["svg"] is None
    finally:
        notebook_code["plt"].close(figure)
