"""Exercise the original jlens adapter on real, tiny HF architectures offline."""

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

import jlens
from jlens.evaluation import (
    DATASETS,
    evaluate_paired,
    identity_lens,
    load_eval,
    logit_lens,
    single_token_ids,
    spelling_ids,
)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/jacobian_lens/model_agnostic_lens_dataset.ipynb"
NOTEBOOKS = [
    NOTEBOOK,
    NOTEBOOK.with_name("failed_gemma_model_agnostic_lens_dataset.ipynb"),
]


class NativeTokenizer:
    """Simulate native tokenizers with and without automatic BOS insertion."""

    all_special_ids = [128]

    def __init__(self, bos):
        self.bos_token_id = 128 if bos else None

    def encode(self, text, *, add_special_tokens=True):
        ids = [ord(char) % 128 for char in text]
        return [128, *ids] if add_special_tokens and self.bos_token_id is not None else ids

    def __call__(self, text, *, max_length=512, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)[:min(max_length, 128)]]))

    def decode(self, ids, **kwargs):
        return "".join(chr(token) if token < 128 else "<BOS>" for token in ids)


@pytest.fixture(params=["gpt2", "qwen2", "llama", "gemma2"])
def native_model(request):
    torch.manual_seed(3)
    common = dict(
        vocab_size=129, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
    )
    if request.param == "gpt2":
        hf = GPT2LMHeadModel(GPT2Config(vocab_size=129, n_positions=128, n_embd=16, n_layer=2, n_head=2))
    elif request.param == "qwen2":
        hf = Qwen2ForCausalLM(Qwen2Config(**common))
    elif request.param == "llama":
        hf = LlamaForCausalLM(LlamaConfig(**common))
    else:
        hf = Gemma2ForCausalLM(Gemma2Config(
            **common, head_dim=8, query_pre_attn_scalar=8, sliding_window=64,
            final_logit_softcapping=0.05,
        ))
    tokenizer = NativeTokenizer(bos=request.param != "qwen2")
    return hf, jlens.from_hf(hf, tokenizer)


def test_handwritten_final_row_is_actual_model_output(native_model):
    hf, model = native_model
    prompt = "abc\ndef"
    logits = logit_lens(model, prompt)
    with torch.no_grad():
        expected = hf(input_ids=model.encode(prompt), use_cache=False).logits[0].float()
    torch.testing.assert_close(logits[-1], expected)
    if model.tokenizer.bos_token_id is not None:
        assert single_token_ids(model.tokenizer, "a") == {ord("a"), ord("A")}
        assert spelling_ids(model.tokenizer, "long") == {ord(" ")}
    # Gemma2's softcap makes skipping the adapter's transform detectable.
    if isinstance(hf, Gemma2ForCausalLM):
        assert logits.abs().max() <= 0.05


def test_paired_metrics_include_first_position_without_bos(native_model):
    _, model = native_model
    samples = [{"name": "one", "prompt": "abc", "intermediates": ["a"], "target": "d"}]
    with patch.object(model, "forward", wraps=model.forward) as forward:
        words, items = evaluate_paired(model, identity_lens(model), {"multihop": samples})
    assert forward.call_count == 1
    for frame in (words, items):
        left = frame[frame.lens == "logit lens"].drop(columns="lens").reset_index(drop=True)
        right = frame[frame.lens == "J-lens"].drop(columns="lens").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
    logits = logit_lens(model, "abc")
    start = int(model.tokenizer.bos_token_id is not None)
    ids = model.encode("abc")[0, start:]
    top1 = logits[:, start:].argmax(-1)
    expected_copy = (top1 == ids).float().mean(-1).numpy()
    np.testing.assert_allclose(items.iloc[0].copy_rate, expected_copy)
    assert words.iloc[0].in_prompt


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("coverage", ["original", "mixed", "unsupported"])
def test_generic_notebook_runs_all_analysis_on_native_adapters(
    native_model, notebook_path, coverage,
):
    hf, model = native_model
    plt.switch_backend("Agg")
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    evals = {dataset: load_eval(str(REPO), dataset)[:2] for dataset in DATASETS}
    if coverage != "original":
        for samples in evals.values():
            for index, sample in enumerate(samples):
                if sample.get("target") is not None:
                    sample["target"] = (
                        "a" if coverage == "mixed" and index == 0
                        else "unsupported multi-token answer"
                    )
    namespace = {
        "model": model, "hf_model": hf, "fitted_lens": identity_lens(model),
        "evals": evals,
        "LENS_NAMES": ("logit lens", "J-lens"), "LAYER_STRIDE": 1,
        "K": 5, "KS": [1, 5, 10, 100], "LAST": model.n_layers - 1,
        "HUB_LENS": None, "loaded_hub_request": None,
    }
    analysis = False
    executed_cells = set()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        patch.object(plt, "show"),
        patch("jlens.metrics.evaluate_distributions") as distributions,
        patch("jlens.metrics.lens_vector_geometry") as geometry,
    ):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            # Imports/comments may precede the identity check. Select the stable
            # cell ID, not source formatting, so all real analysis cells execute.
            if cell.id == "generic-014":
                analysis = True
            if cell.id in ("generic-004", "generic-008", "generic-010") or analysis:
                exec(compile(source, f"{notebook_path.name}:cell {index}", "exec"), namespace)
                executed_cells.add(cell.id)
                plt.close("all")
    assert {"generic-014", "generic-016", "generic-055"} <= executed_cells
    assert len(namespace["model_items"]) == 12
    assert len(namespace["head_to_head"]) == len(namespace["inter"]) // 2
    assert not any("GPT2LensModel" in cell.source for cell in notebook.cells)
    distributions.assert_not_called()
    geometry.assert_not_called()
    assert namespace["held_out_metrics"] is None
    assert namespace["fig55"] is namespace["fig56"] is None
    assert namespace["reference52"].counts is not None
    assert namespace["words"].ranks.notna().all()
    assert not namespace["unsupported_targets"].empty
    answer_coverage = namespace["answer_coverage"]
    assert (answer_coverage.scored + answer_coverage.unsupported == answer_coverage.total).all()
    if coverage == "unsupported":
        assert (answer_coverage.coverage == 0).all()
        assert namespace["accuracy"].empty
        assert namespace["target"].empty
        assert any("accuracy n/a" in line for line in namespace["lines"])
    elif coverage == "mixed":
        assert (answer_coverage.coverage == 0.5).all()
    # The top-1 display table must not shadow the dual-readout callback's helper.
    # Rerun must recompute, not reuse the previous metrics or held-out results.
    cells = {cell.id: cell.source for cell in notebook.cells}
    old_words = namespace["words"]
    namespace["held_out_metrics"] = object()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        patch.object(model, "forward", wraps=model.forward) as forward,
    ):
        exec(compile(cells["generic-016"], notebook_path.name, "exec"), namespace)
    assert forward.call_count == sum(map(len, evals.values()))
    assert namespace["words"] is not old_words
    assert namespace["held_out_metrics"] is None
    assert namespace["accuracy"] is namespace["reference52"] is None


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
@pytest.mark.parametrize("failure", ["configuration", "inference"])
def test_notebook_failed_evaluation_invalidates_previous_tables(notebook_path, failure):
    cells = {c.id: c.source for c in nbformat.read(notebook_path, as_version=4).cells}
    keys = (
        "words", "items", "all_words", "scores", "inter", "target", "accuracy",
        "answer_coverage", "word_coverage", "paired_summary", "reference52",
        "held_out_metrics", "held_out_geometry", "fig55", "fig56",
    )
    ns = dict.fromkeys(keys, "old result")
    ns.update(fitted_lens=None if failure == "configuration" else object(),
              HUB_LENS=None, loaded_hub_request=None, model=object(), evals={},
              handwritten_logit_lens=object())

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    ns["evaluate_paired"] = fail
    with pytest.raises(RuntimeError, match="configuration|inference"):
        exec(compile(cells["generic-016"], notebook_path.name, "exec"), ns)
    assert all(ns[key] is None for key in keys)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.stem)
def test_generic_reference_opt_in_and_cached_plots(native_model, notebook_path):
    _, model = native_model
    plt.switch_backend("Agg")
    cells = {c.id: c.source for c in nbformat.read(notebook_path, as_version=4).cells}
    lens = identity_lens(model)
    words, _ = evaluate_paired(model, lens, {
        "multihop": [{"name": "one", "prompt": "abc", "intermediates": ["a"]}],
    })
    ns = {"model": model, "fitted_lens": lens, "words": words,
          "plt": plt, "torch": torch, "display": lambda *args: None}

    def run(cell):
        exec(compile(cells[cell], cell, "exec"), ns)

    with patch.object(plt, "show"), patch.object(model, "forward", wraps=model.forward) as forward:
        run("reference-52")
        run("reference-heldout-config")
        run("reference-heldout-evaluate")
        run("reference-heldout-plot")
        forward.assert_not_called()
        ns["HELD_OUT_TEXTS"] = ["held out one", "held out two"]
        with pytest.raises(ValueError, match="HELD_OUT_SOURCE"):
            run("reference-heldout-evaluate")
        forward.assert_not_called()
        ns["HELD_OUT_SOURCE"] = "independent synthetic strings"
        run("reference-heldout-evaluate")
        assert forward.call_count == 2
        assert len(ns["held_out_metrics"].layers) == 2 * model.n_layers
        assert set(ns["held_out_geometry"].status) == {"ok"}
        run("reference-heldout-plot")
        run("reference-heldout-plot")
        assert forward.call_count == 2
        assert len(ns["fig55"].axes) == len(ns["fig56"].axes) == 3
        # An arbitrary LensModel may decode correctly without exposing a linear head.
        ns["model"] = SimpleNamespace(**{
            name: getattr(model, name) for name in (
                "layers", "n_layers", "d_model", "tokenizer", "input_device",
                "encode", "forward", "unembed",
            )
        })
        run("reference-heldout-evaluate")
        assert set(ns["held_out_geometry"].status) == {"unsupported"}
        assert ns["held_out_geometry"].mean_cosine.isna().all()
        run("reference-heldout-plot")
        assert ns["axes56"][0].get_ylim() == (0, 1)
        run("reference-heldout-config")
        run("reference-heldout-evaluate")
        run("reference-heldout-plot")
        assert ns["held_out_metrics"] is None
        assert ns["fig55"] is ns["fig56"] is None
    plt.close("all")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_generic_identity_uses_configured_dtype_without_casting(native_model, dtype):
    hf, model = native_model
    hf.to(dtype=dtype)
    cells = {c.id: c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells}
    ns = {"model": model, "hf_model": hf,
          "evals": {"order-ops": [{"name": "one", "prompt": "abc", "intermediates": ["a"]}]}}
    for cell in ("generic-004", "generic-008", "generic-010", "generic-014"):
        exec(compile(cells[cell], cell, "exec"), ns)
    assert {p.dtype for p in hf.parameters()} == {dtype}
