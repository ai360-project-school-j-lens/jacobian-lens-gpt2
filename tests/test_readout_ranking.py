"""Offline saturation regressions for lexical scores and model distributions."""

import importlib
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from jlens import ActivationRecorder, LensReadout, from_hf
from jlens.batched_evaluation import (
    evaluate_distributions_batched,
    evaluate_paired_batched,
)
from jlens.evaluation import (
    evaluate_paired,
    evaluate_readout,
    identity_lens,
    jacobian_lens,
    lens_slice,
    logit_lens,
    logit_lens_from_activations,
    token_ranks,
    top_tokens_table,
)
from jlens.metrics import evaluate_distributions
from jlens.readout import readout, selected_token_ranks, top_token_ids
from jlens.readout_cache import WordRanker
from jlens.vis import _ranks_of, compute_slice


class Tokenizer:
    all_special_ids = []
    bos_token_id = eos_token_id = pad_token_id = None

    def encode(self, text, **kwargs):
        return [ord(c) - ord("a") for c in text.lower().strip()]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)]))

    def decode(self, ids, **kwargs):
        return "".join(chr(ord("a") + i) for i in ids)


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, h):
        return h * self.scale


class Decoder(nn.Module):
    def __init__(self, final_scale):
        super().__init__()
        self.embed_tokens = nn.Embedding(5, 1)
        nn.init.ones_(self.embed_tokens.weight)
        self.layers = nn.ModuleList([Scale(1), Scale(final_scale)])
        self.norm = nn.Identity()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(last_hidden_state=h)


class CausalLM(nn.Module):
    def __init__(self, final_scale):
        super().__init__()
        self.model = Decoder(final_scale)
        self.lm_head = nn.Linear(1, 5, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[90], [100], [110], [150], [300]]))
        self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(
            num_hidden_layers=2, hidden_size=1, final_logit_softcapping=30,
        ))

    def forward(self, input_ids):
        raw = self.lm_head(self.model(input_ids).last_hidden_state)
        return SimpleNamespace(logits=30 * torch.tanh(raw / 30))


def model_with_saturation(final_scale=1):
    return from_hf(CausalLM(final_scale).to(torch.bfloat16), Tokenizer())


def lookup(word, expand=False):
    return {ord(word) - ord("a")}


def samples():
    return {"multihop": [
        {"name": "one", "prompt": "ab", "intermediates": ["c"], "target": "e"},
        {"name": "two", "prompt": "abc", "intermediates": ["d"], "target": "c"},
    ]}


def test_native_softcap_is_unchanged_but_lexical_order_survives():
    model = model_with_saturation()
    result = readout(model, torch.ones(1, 1))
    assert result.logits.dtype == result.ranking_scores.dtype == torch.bfloat16
    assert result.logits.tolist() == [[29.875, 29.875, 30, 30, 30]]
    assert result.logits.argmax(-1).item() == 2
    assert result.ranking_scores.argmax(-1).item() == 4
    assert token_ranks(result.ranking_scores, set(range(5))).tolist() == [[5, 4, 3, 2, 1]]
    # Legacy tensors no longer give every saturated maximum a top-1 hit either.
    assert token_ranks(result.logits, set(range(5))).tolist() == [[4, 5, 1, 2, 3]]
    torch.testing.assert_close(model.unembed(torch.ones(1, 1)), result.logits)
    actual = model._hf_model(model.encode("ab")).logits[0].float()
    torch.testing.assert_close(logit_lens(model, "ab")[-1], actual)
    dual = logit_lens(model, "ab", return_readout=True)
    transported = jacobian_lens(model, identity_lens(model), "ab", return_readout=True)
    torch.testing.assert_close(dual.logits, transported.logits)
    torch.testing.assert_close(dual.ranking_scores, transported.ranking_scores)
    assert isinstance(logit_lens(model, "ab"), torch.Tensor)
    assert isinstance(jacobian_lens(model, identity_lens(model), "ab"), torch.Tensor)


def test_fp32_softcap_also_saturates_and_exact_ties_have_one_order():
    scores = torch.tensor([[1000., 2000., 3000., 3000., -1000.]])
    assert (30 * torch.tanh(scores / 30)).tolist() == [[30, 30, 30, 30, -30]]
    expected = [[4, 3, 1, 2, 5]]
    assert token_ranks(scores, set(range(5))).tolist() == expected
    assert top_token_ids(scores, 5).tolist() == [[2, 3, 1, 0, 4]]
    np.testing.assert_array_equal(_ranks_of(scores, torch.arange(5)) + 1, expected)
    ids = torch.tensor([[0, 3], [4, 2]])
    rows = scores.expand(2, -1)
    assert selected_token_ranks(rows, ids).tolist() == [[4, 2], [5, 1]]
    assert token_ranks(scores, set()).shape == (1, 0)


@pytest.mark.parametrize("final_scale", [1, 0.1])
def test_shared_batched_and_single_readouts_keep_distribution_metrics(final_scale):
    model = model_with_saturation(final_scale)
    lens = identity_lens(model)
    evals = samples()
    words, items = evaluate_paired(model, lens, evals, spelling_lookup=lookup)
    with patch.object(model, "forward", wraps=model.forward) as forward:
        batched_words, batched_items = evaluate_paired_batched(
            model, lens, evals, spelling_lookup=lookup, bos_policy="none",
            position_chunk_size=2, rank_chunk_size=1, batch_size=2, progress=False,
        )
    assert forward.call_count == 1  # Mixed lengths, one padded model pass.
    for chunk in (1, 8):
        w, i = evaluate_paired_batched(
            model, lens, evals, spelling_lookup=lookup, bos_policy="none",
            position_chunk_size=chunk, rank_chunk_size=chunk, progress=False,
            batching="equal_length",
        )
        pd.testing.assert_frame_equal(w, batched_words)
        pd.testing.assert_frame_equal(i, batched_items)
    pd.testing.assert_frame_equal(words, batched_words[words.columns])
    metrics = ["agreement", "copy_rate", "kl_to_final"]
    pd.testing.assert_frame_equal(items.drop(columns=metrics), batched_items[items.columns].drop(columns=metrics))
    for column in metrics:
        np.testing.assert_allclose(np.stack(items[column]), np.stack(batched_items[column]), atol=1e-6)
    assert words.iloc[0].ranks[0] == 3  # Saturated "c" is NOT lexical top-1.
    assert items.iloc[0].readout_top1[0] == "c"  # Actual distribution top-1.
    result = logit_lens(model, "ab", return_readout=True)
    logp = result.logits.float().log_softmax(-1)
    expected_kl = (logp[-1].exp() * (logp[-1] - logp)).sum(-1).mean(-1)
    np.testing.assert_allclose(items.iloc[0].kl_to_final, expected_kl, atol=1e-6)
    assert items.iloc[0].kl_to_final[-1] == 0
    if final_scale == 1:
        assert batched_items.iloc[0].model_top1_id == 2
        assert batched_items.iloc[0].model_prediction_ranks[-1] == 3
        assert not items.iloc[0].model_correct  # Actual argmax != lexical top-1.
    else:
        assert items.iloc[0].kl_to_final[0] > 0
    one_words, one_items = evaluate_readout(
        model, lambda text: logit_lens(model, text, return_readout=True),
        "multihop", evals["multihop"],
    )
    baseline = words[words.lens == "logit lens"].drop(columns="lens").reset_index(drop=True)
    pd.testing.assert_frame_equal(one_words, baseline)
    baseline_items = items[items.lens == "logit lens"].drop(columns="lens").reset_index(drop=True)
    pd.testing.assert_frame_equal(one_items, baseline_items)


def test_transported_scores_and_final_identity_are_shared():
    model = model_with_saturation()
    lens = identity_lens(model)
    lens.jacobians[0] = -torch.eye(1)
    lens.jacobians[1] = -torch.eye(1)  # Final-layer fits must never alter evaluation.
    words, _ = evaluate_paired(model, lens, samples(), spelling_lookup=lookup)
    batched, _ = evaluate_paired_batched(
        model, lens, samples(), spelling_lookup=lookup, bos_policy="none", progress=False,
    )
    pd.testing.assert_frame_equal(words, batched[words.columns])
    d_rows = words[(words.word == "d") & (words.item == "one")]
    assert d_rows.iloc[0].ranks.tolist() == [2, 2]
    assert d_rows.iloc[1].ranks.tolist() == [4, 2]
    result = jacobian_lens(model, lens, "ab", return_readout=True)
    baseline = logit_lens(model, "ab", return_readout=True)
    torch.testing.assert_close(result.logits[-1], baseline.logits[-1])
    torch.testing.assert_close(result.ranking_scores[-1], baseline.ranking_scores[-1])


def test_exact_head_ties_and_multiple_spellings_agree_across_rankers():
    model = model_with_saturation()
    with torch.no_grad():
        model._lm_head.weight.copy_(torch.tensor([[90], [110], [110], [300], [300]]))

    def tied_lookup(word, expand=False):
        return {2, 4} if word == "c" else {3, 4}

    words, _ = evaluate_paired(model, identity_lens(model), samples(), spelling_lookup=tied_lookup)
    batched, _ = evaluate_paired_batched(
        model, identity_lens(model), samples(), spelling_lookup=tied_lookup,
        bos_policy="none", rank_chunk_size=1, progress=False,
    )
    pd.testing.assert_frame_equal(words, batched[words.columns])
    assert words.iloc[0].ranks.tolist() == [2, 2]  # ID 3 precedes equally scored ID 4.
    scores = readout(model, torch.ones(1, 1))
    ranker = WordRanker(pd.DataFrame({"item_index": [0, 0], "ids": [[2, 4], [4, 3]]}))
    np.testing.assert_array_equal(ranker(scores), [2, 1])
    assert top_token_ids(scores.ranking_scores, 5).tolist() == [[3, 4, 1, 2, 0]]


def test_legacy_custom_callbacks_warn_and_remain_identity_comparable():
    model = model_with_saturation()
    with pytest.warns(UserWarning, match="Tensor-only readout"):
        words, items = evaluate_paired(
            model, identity_lens(model), samples(),
            logit_readout=logit_lens_from_activations, spelling_lookup=lookup,
        )
    for frame in (words, items):
        a = frame[frame.lens == "logit lens"].drop(columns="lens").reset_index(drop=True)
        b = frame[frame.lens == "J-lens"].drop(columns="lens").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    assert words.iloc[0].ranks.tolist() == [1, 1]  # Legacy distribution ordering.
    with pytest.warns(UserWarning, match="Tensor-only readout"):
        evaluate_readout(model, lambda text: logit_lens(model, text), "multihop", samples()["multihop"])
    dual_words, _ = evaluate_paired(
        model, identity_lens(model), samples(), spelling_lookup=lookup,
        logit_readout=lambda m, a: logit_lens_from_activations(m, a, return_readout=True),
    )
    assert dual_words.iloc[0].ranks.tolist() == [3, 3]


def test_cache_and_displays_use_the_same_pre_softcap_ranks():
    model = model_with_saturation()
    def lens_fn(text):
        return logit_lens(model, text, return_readout=True)

    result = lens_fn("ab")
    words = pd.DataFrame({"item_index": [0, 0], "ids": [[2], [3, 4]]})
    ranker = WordRanker(words)
    values = LensReadout(result.logits[:, :1], result.ranking_scores[:, :1])
    np.testing.assert_array_equal(ranker(values, row_chunk=1), [[3, 1], [3, 1]])
    np.testing.assert_array_equal(
        ranker.rank_residuals(lambda h: readout(model, h), torch.ones(2, 1, 1), batch=1),
        [[3, 1], [3, 1]],
    )
    table = top_tokens_table(model, {"lexical": lens_fn}, "ab", top_n=3)
    assert table.iloc[0, 0] == ["e", "d", "c"]
    shared = lens_slice(model, lens_fn, "ab", top_n=3)
    direct = compute_slice(model, identity_lens(model), "ab", top_n=3)
    np.testing.assert_array_equal(shared.top_ids, direct.top_ids)
    np.testing.assert_array_equal(shared.rank_tensor, direct.rank_tensor)
    assert shared.top_ids[0, 0].tolist() == [4, 3, 2]


def test_distribution_evaluators_use_capped_logits_not_lexical_scores():
    model = model_with_saturation(0.1)
    lens = identity_lens(model)
    texts = ["ab", "abc"]
    serial = evaluate_distributions(model, lens, texts, progress=False)
    batched = evaluate_distributions_batched(model, lens, texts, progress=False)
    pd.testing.assert_frame_equal(serial.layers, batched.layers, atol=1e-6)
    pd.testing.assert_frame_equal(serial.pairs, batched.pairs, atol=1e-6)
    result = logit_lens(model, "ab", return_readout=True)
    logp = result.logits.float().log_softmax(-1)
    kl = (logp[-1].exp() * (logp[-1] - logp)).sum(-1).mean(-1)
    entropy = -(logp.exp() * logp).sum(-1).mean(-1)
    for name in ("logit lens", "J-lens"):
        rows = serial.layers[serial.layers.lens == name]
        np.testing.assert_allclose(rows.kl_model_to_lens, kl, atol=1e-6)
        np.testing.assert_allclose(rows.entropy, entropy, atol=1e-6)
        assert rows.top1_agreement.tolist() == [0, 1]


def test_old_adapters_and_nonsoftcapped_hf_keep_their_contract():
    residual = torch.ones(2, 1)
    legacy = SimpleNamespace(unembed=lambda h: h * 2)
    result = readout(legacy, residual)
    assert result.logits is result.ranking_scores
    model = model_with_saturation()
    model._logit_softcap = None
    result = readout(model, residual)
    assert result.logits is result.ranking_scores
    torch.testing.assert_close(result.logits, model.unembed(residual))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_chunked_topk_is_compact_and_matches_stable_order(monkeypatch, dtype):
    module = importlib.import_module("jlens.readout")
    monkeypatch.setattr(module, "_ROW_CHUNK", 2)
    monkeypatch.setattr(module, "_VOCAB_CHUNK", 7)
    # Noncontiguous, with ties spanning every vocabulary chunk and row batch.
    scores = (torch.arange(3 * 4 * 29).reshape(3, 4, 29) % 5).to(dtype).transpose(0, 1)
    expected = scores.argsort(dim=-1, descending=True, stable=True)
    original = torch.Tensor.argsort
    sort_shapes = []

    def bounded_sort(tensor, *args, **kwargs):
        sort_shapes.append(tensor.shape)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "argsort", bounded_sort)
    top = top_token_ids(scores, 3)
    torch.testing.assert_close(top, expected[..., :3])
    assert top.untyped_storage().nbytes() == top.numel() * top.element_size()
    assert all(s[0] <= 2 and s[-1] <= 7 for s in sort_shapes)
    for k in (0, 1, 10, 29):
        result = top_token_ids(scores, k)
        torch.testing.assert_close(result, expected[..., :k])
        assert result.untyped_storage().nbytes() == result.numel() * result.element_size()
    torch.testing.assert_close(top_token_ids(torch.zeros(20), 4), torch.arange(4))


def test_chunked_ranks_match_inverse_order_and_index_rows_without_vocab_copy(monkeypatch):
    module = importlib.import_module("jlens.readout")
    monkeypatch.setattr(module, "_ROW_CHUNK", 2)
    monkeypatch.setattr(module, "_VOCAB_CHUNK", 5)
    monkeypatch.setattr(module, "_TARGET_CHUNK", 3)
    scores = (torch.arange(4 * 3 * 23).reshape(4, 3, 23) % 7).double().transpose(0, 1)
    order = scores.argsort(dim=-1, descending=True, stable=True)
    expected = order.argsort(dim=-1) + 1
    ids = torch.arange(23)
    torch.testing.assert_close(selected_token_ranks(scores, ids), expected)
    assert selected_token_ranks(scores, ids[:0]).shape == (*scores.shape[:-1], 0)
    torch.testing.assert_close(selected_token_ranks(scores[0, 0], ids), expected[0, 0])
    rows = torch.tensor([3, 0, 3, 2, 1])
    targets = torch.tensor([[0, 5, 22], [2, 8, 11], [4, 6, 1], [3, 9, 7], [7, 8, 9]])
    # Reject any advanced-index gather of an entire vocabulary row.
    original = torch.Tensor.__getitem__

    def bounded_getitem(tensor, index):
        if isinstance(index, torch.Tensor) and tensor.ndim == 2:
            assert tensor.shape[-1] <= 5
        return original(tensor, index)

    reference = expected[0, rows].gather(-1, targets)
    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "__getitem__", bounded_getitem)
        actual = selected_token_ranks(scores[0], targets, row_indices=rows)
    torch.testing.assert_close(actual, reference)
    assert selected_token_ranks(scores[0], targets[:0], row_indices=rows[:0]).shape == (0, 3)


@pytest.mark.parametrize("callback_kind", ["tensor", "dual", "aliased_dual"])
def test_paired_does_not_clone_or_mutate_callback_storage(monkeypatch, callback_kind):
    model = model_with_saturation()
    # Expanded views intentionally overlap across layers and positions.
    logits = torch.tensor([90., 100., 110., 150., 300.]).expand(2, 2, 5)
    scores = logits if callback_kind != "dual" else -logits
    value = logits if callback_kind == "tensor" else LensReadout(logits, scores)
    before_logits, before_scores = logits.clone(), scores.clone()
    original = torch.Tensor.clone

    def no_readout_clone(tensor, *args, **kwargs):
        assert tensor.shape != logits.shape
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", no_readout_clone)
    with pytest.warns(UserWarning) if callback_kind == "tensor" else nullcontext():
        _, items = evaluate_paired(
            model, identity_lens(model), {"multihop": samples()["multihop"][:1]},
            logit_readout=lambda m, a: value, spelling_lookup=lookup,
        )
    torch.testing.assert_close(logits, before_logits)
    torch.testing.assert_close(scores, before_scores)
    assert items.iloc[0].readout_top1 == ["e", "c"]
    assert items.iloc[0].kl_to_final[-1] == 0


def test_batched_and_cached_rankers_pass_indices_instead_of_copied_rows(monkeypatch):
    calls = []

    def indexed_rank(scores, ids, *, row_indices=None):
        assert row_indices is not None
        calls.append(scores.shape)
        return selected_token_ranks(scores, ids, row_indices=row_indices)

    for name in ("jlens.batched_evaluation", "jlens.readout_cache"):
        monkeypatch.setattr(importlib.import_module(name), "selected_token_ranks", indexed_rank)
    model = model_with_saturation()
    evaluate_paired_batched(
        model, identity_lens(model), samples(), spelling_lookup=lookup,
        bos_policy="none", position_chunk_size=2, progress=False,
    )
    assert calls and all(shape[0] <= 2 for shape in calls)
    calls.clear()
    ranker = WordRanker(pd.DataFrame({"item_index": [0] * 100, "ids": [[4]] * 100}))
    np.testing.assert_array_equal(ranker(readout(model, torch.ones(1, 1))), np.ones(100))
    assert calls == [torch.Size([1, 5])]  # Not a [100, vocab] word expansion.


def test_default_paired_streams_layers_without_full_stacks(monkeypatch):
    module = importlib.import_module("jlens.evaluation")
    model = model_with_saturation()
    model.layers.extend([Scale(1), Scale(1)])
    model.n_layers = 4
    original = module.readout
    previous = []

    def streaming_readout(*args):
        # Only the shared final readout can survive between layer decodes.
        assert sum(ref() is not None for ref in previous) <= 1
        result = original(*args)
        previous.append(weakref.ref(result))
        return result

    def forbidden_stack(*args, **kwargs):
        pytest.fail("paired evaluation must not stack full vocabulary readouts")

    monkeypatch.setattr(module, "readout", streaming_readout)
    monkeypatch.setattr(module, "_stack_readouts", forbidden_stack)
    evaluate_paired(model, identity_lens(model), samples(), spelling_lookup=lookup)
    assert len(previous) == len(samples()["multihop"]) * (1 + 2 * (model.n_layers - 1))
    assert all(ref() is None for ref in previous)


@pytest.mark.parametrize("family", ["gemma2", "gemma4"])
def test_real_tiny_gemma_bf16_distribution_matches_hf(family):
    import transformers

    torch.manual_seed(9)
    common = dict(
        vocab_size=5, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, sliding_window=16, final_logit_softcapping=0.01,
    )
    if family == "gemma2":
        config = transformers.Gemma2Config(**common, query_pre_attn_scalar=8)
        hf = transformers.Gemma2ForCausalLM(config)
    else:
        config = transformers.Gemma4TextConfig(
            **common, global_head_dim=8, hidden_size_per_layer_input=0,
            num_kv_shared_layers=0, layer_types=["sliding_attention", "full_attention"],
        )
        hf = transformers.Gemma4ForCausalLM(config)
    model = from_hf(hf.to(torch.bfloat16), Tokenizer())
    ids = model.encode("bcd")
    with torch.no_grad(), ActivationRecorder(model.layers, at=[1]) as recorder:
        expected = hf(input_ids=ids, use_cache=False).logits
    result = readout(model, recorder.activations[1])
    torch.testing.assert_close(result.logits.float(), expected.float(), rtol=0, atol=0)
    raw = model._lm_head(model._final_norm(recorder.activations[1]))
    torch.testing.assert_close(result.ranking_scores, raw, rtol=0, atol=0)
    assert len(torch.unique(raw)) > len(torch.unique(result.logits))
    assert {p.dtype for p in hf.parameters()} == {torch.bfloat16}
