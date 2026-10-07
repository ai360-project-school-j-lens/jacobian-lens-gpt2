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


def test_generic_notebook_runs_all_analysis_on_native_adapters(native_model):
    hf, model = native_model
    plt.switch_backend("Agg")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    namespace = {
        "model": model, "hf_model": hf, "fitted_lens": identity_lens(model),
        "evals": {dataset: load_eval(str(REPO), dataset)[:2] for dataset in DATASETS},
        "LENS_NAMES": ("logit lens", "J-lens"), "LAYER_STRIDE": 1,
        "K": 5, "KS": [1, 5, 10, 100], "LAST": model.n_layers - 1,
    }
    analysis = False
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch.object(plt, "show"):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            source = cell.source
            if source.startswith("sanity_evals ="):
                analysis = True
            if (source.startswith("import matplotlib") or source.startswith("@torch.no_grad()")
                    or source.startswith("check_prompt =") or analysis):
                exec(compile(source, f"{NOTEBOOK.name}:cell {index}", "exec"), namespace)
                plt.close("all")
    assert len(namespace["model_items"]) == 12
    assert len(namespace["head_to_head"]) == len(namespace["inter"]) // 2
    assert not any("GPT2LensModel" in cell.source for cell in notebook.cells)
