"""Shared streaming evaluation preserves full tables and callback contracts."""

import ast
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import nbformat
import pandas as pd
import pytest
import torch

from jlens.evaluation import _readout_rows, evaluate_paired, identity_lens
from jlens.layerwise_evaluation import evaluate_paired_layerwise
from jlens.readout import LensReadout, readout
from tests.test_generic_evaluation import NOTEBOOKS, NativeTokenizer
from tests.test_generic_evaluation import native_model as native_model
from tests.tiny import TinyDecoder


def _callback(notebook_path):
    cells = {c.id: c.source for c in nbformat.read(notebook_path, as_version=4).cells}
    ns = {}
    exec(cells["generic-004"], ns)
    exec(cells["generic-008"], ns)
    return ns


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_layerwise_matches_full_tables(native_model, notebook_path, dtype):
    hf, model = native_model
    hf.to(dtype=dtype)
    ns = _callback(notebook_path)
    callback = ns["handwritten_logit_lens"]
    lens = identity_lens(model)
    # Nonidentity transports check more than the shared-final/identity cases.
    lens.jacobians[0] *= 1.7
    evals = {
        "multihop": [
            {"name": "one", "prompt": "abc ", "intermediates": ["a", "long"], "target": "d"},
            {"name": "two", "prompt": "bc\nde", "intermediates": ["b"], "target": "long"},
        ],
        "poetry": [{"name": "verse", "prompt": "abc\ndef", "intermediates": ["d"], "target": "f"}],
    }
    expected = evaluate_paired(model, lens, evals, logit_readout=callback)
    with (
        patch.object(model, "forward", wraps=model.forward) as forward,
        patch("jlens.evaluation._readout_rows", wraps=_readout_rows) as rows,
    ):
        actual = ns["evaluate_paired"](model, lens, evals)
    assert forward.call_count == 3
    assert rows.call_count == 6  # One scoring pass per lens, not per layer.
    assert all(callable(call.args[1]) for call in rows.call_args_list)
    selected_layers = []

    def selected_callback(model, activations, *, layers):
        selected_layers.append(layers)
        result = callback(model, activations, layers=layers)
        assert result.logits.shape[0] == 1
        assert result.ranking_scores.dtype == dtype
        return result

    compatible = evaluate_paired_layerwise(model, lens, evals, logit_readout=selected_callback)
    assert selected_layers == [[layer] for _ in range(3) for layer in range(model.n_layers - 1)]
    for result in (actual, compatible):
        for left, right in zip(result, expected, strict=True):
            pd.testing.assert_frame_equal(left, right, check_exact=True)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_notebook_selected_layer_retains_precision_and_avoids_stack(notebook_path, dtype):
    ns = _callback(notebook_path)
    model = SimpleNamespace(n_layers=60)
    scores = torch.tensor([[1., 1. + 2**-40, 2.]], dtype=dtype)
    logits = scores.tanh()
    value = LensReadout(logits, scores)
    with patch.dict(ns, lens_readout=lambda model, h: value):
        result = ns["handwritten_logit_lens"](model, {59: torch.zeros(1, 1, 8)}, layers=[59])
    assert result.ranking_scores.dtype == dtype
    assert result.ranking_scores.data_ptr() == scores.data_ptr()
    assert result.logits.data_ptr() == logits.data_ptr()
    torch.testing.assert_close(result.ranking_scores[0], scores, rtol=0, atol=0)


@pytest.mark.parametrize("n_layers", [1, 8])
def test_workspace_layer_bound_does_not_grow_with_model_depth(n_layers):
    model = TinyDecoder(n_layers=n_layers, vocab_size=129)
    model.tokenizer = NativeTokenizer(bos=False)
    lens = identity_lens(model)
    evals = {"multihop": [{"name": "one", "prompt": "abc", "intermediates": ["a"]}]}
    with patch("jlens.evaluation._readout_rows", wraps=_readout_rows) as rows:
        actual = evaluate_paired_layerwise(model, lens, evals)
    assert rows.call_count == 2
    assert all(callable(call.args[1]) for call in rows.call_args_list)
    expected = evaluate_paired(model, lens, evals)
    for left, right in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_one_layer_tensor_readout_uses_its_own_depth():
    model = TinyDecoder(n_layers=8, vocab_size=129)
    model.tokenizer = NativeTokenizer(bos=False)
    samples = [{"name": "one", "prompt": "abc", "intermediates": ["a"]}]
    ids = model.encode("abc")[0].tolist()
    logits = torch.zeros(1, len(ids), 129)
    words, item = _readout_rows(model, LensReadout(logits, logits), "multihop", samples, 0, ids)
    assert words[0]["ranks"].shape == (1,)
    assert item["agreement"].tolist() == [1.0]
    assert item["kl_to_final"].tolist() == [0.0]


def test_layer_callback_storage_is_released_and_not_mutated():
    model = TinyDecoder(n_layers=8, vocab_size=129)
    model.tokenizer = NativeTokenizer(bos=False)
    lens = identity_lens(model)
    evals = {"multihop": [{"name": "one", "prompt": "abc", "intermediates": ["a"]}]}
    references = []
    calls = []

    def callback(model, activations, *, layers):
        assert all(ref() is None for ref in references)
        assert len(layers) == 1
        calls.extend(layers)
        value = readout(model, activations[layers[0]][0].float())
        # Views must not be cloned or kept alive after this layer is scored.
        result = LensReadout(value.logits[None], value.ranking_scores[None])
        references.extend([weakref.ref(result.logits), weakref.ref(result.ranking_scores)])
        return result

    expected = evaluate_paired(model, lens, evals)
    actual = evaluate_paired_layerwise(model, lens, evals, logit_readout=callback)
    assert calls == list(range(model.n_layers - 1))
    assert all(ref() is None for ref in references)
    for left, right in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)


@pytest.mark.parametrize("bad", ["tensor", "shape", "both"])
def test_callback_contract_errors(bad):
    model = TinyDecoder(n_layers=2, vocab_size=129)
    model.tokenizer = NativeTokenizer(bos=False)
    evals = {"multihop": [{"name": "one", "prompt": "abc", "intermediates": ["a"]}]}
    value = torch.zeros(2, 3, 129)

    def callback(*args, **kwargs):
        return value if bad == "tensor" else LensReadout(value, value)

    with pytest.raises(TypeError if bad == "tensor" else ValueError, match="readout"):
        evaluate_paired(
            model, identity_lens(model), evals, layer_logit_readout=callback,
            logit_readout=callback if bad == "both" else None,
        )


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_production_uses_default_streaming(notebook_path):
    cells = {c.id: c.source for c in nbformat.read(notebook_path, as_version=4).cells}
    ns = _callback(notebook_path)
    assert ns["evaluate_paired"] is evaluate_paired
    calls = [node for node in ast.walk(ast.parse(cells["generic-016"]))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "evaluate_paired"]
    assert len(calls) == 1
    assert not calls[0].keywords


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_final_check_only_unembeds_one_layer(native_model, notebook_path):
    hf, model = native_model
    ns = _callback(notebook_path)
    ns.update(model=model, hf_model=hf, evals={"order-ops": [{"prompt": "abc"}]})
    cells = {c.id: c.source for c in nbformat.read(notebook_path, as_version=4).cells}
    with patch.dict(ns, lens_readout=ns["lens_readout"]):
        with patch("jlens.readout.readout", wraps=ns["lens_readout"]) as read:
            ns["lens_readout"] = read
            exec(cells["generic-010"], ns)
    assert read.call_count == 1
