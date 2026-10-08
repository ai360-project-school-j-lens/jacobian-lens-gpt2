"""Offline answer acceptance checks, independent of rank implementation details."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from jlens.batched_evaluation import evaluate_paired_batched
from jlens.evaluation import (
    _readout_rows,
    evaluate_paired,
    evaluate_readout,
    identity_lens,
    single_token_ids,
    spelling_ids,
)
from jlens.strict_scoring import DecodedSpellings
from tests.tiny import TinyDecoder


class AnswerTokenizer:
    """Whole tokens, split answers, unknowns, and duplicate decoded spellings."""

    all_special_ids = [0]
    bos_token_id = None
    pieces = [
        "<unk>", " ", "New", " New", "York", " York", "prompt",
        " 21", " twenty-one", "Twenty-one", "+", " plus", "wrong",
        " Café", " twenty-one", "", "\ufffd", " \ttwenty-one", "  twenty-one",
    ]

    def __init__(self, *, separate_space=False):
        self.separate_space = separate_space

    def get_vocab(self):
        # Vocabulary marker strings deliberately differ from decoded spellings.
        return {f"piece_{i}": i for i in range(len(self.pieces))}

    def encode(self, text, *, add_special_tokens=True):
        if text in {"unknown", "Unknown", " unknown", " Unknown"}:
            return [0]
        if text in {"New York", "new york", " New York", " new york"}:
            if text.startswith(" "):
                return [1, 2, 5] if self.separate_space else [3, 5]
            return [2, 5]
        if text in self.pieces:
            return [self.pieces.index(text)]
        return [1, 12]  # No complete token for other strings/case variants.

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[int(i)] for i in ids)


class AnswerModel(TinyDecoder):
    """Actual CPU activation hooks with controlled final answer logits."""

    def __init__(self, prediction, *, separate_space=False):
        super().__init__(n_layers=2, vocab_size=len(AnswerTokenizer.pieces))
        self.tokenizer = AnswerTokenizer(separate_space=separate_space)
        self.prediction = prediction
        self.eval()

    def encode(self, text, *, max_length=512):
        return torch.tensor([self.tokenizer.encode(text)])

    def unembed(self, residual):
        logits = residual.new_zeros(*residual.shape[:-1], len(self.tokenizer.pieces))
        logits[..., self.prediction] = 5
        return logits


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def evaluate(model, api, target, dataset="multihop"):
    samples = [{"name": "answer", "prompt": "prompt", "intermediates": []}]
    if target is not None:
        samples[0]["target"] = target
    if api == "readout":
        logits = model.unembed(torch.zeros(2, 1, model.d_model))
        return evaluate_readout(model, lambda _: logits, dataset, samples)
    lookup = DecodedSpellings(model.tokenizer, vocab_size=len(model.tokenizer.pieces))
    kwargs = {} if api == "paired" else {"spelling_lookup": lookup}
    if api == "batched":
        return evaluate_paired_batched(
            model, identity_lens(model), {dataset: samples},
            bos_policy="none", progress=False, **kwargs,
        )
    return evaluate_paired(model, identity_lens(model), {dataset: samples}, **kwargs)


APIS = ["readout", "paired", "strict", "batched"]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("separate_space,prediction", [(True, 1), (False, 3)])
def test_multi_token_answers_do_not_accept_whitespace_or_prefix(
    api, separate_space, prediction,
):
    model = AnswerModel(prediction, separate_space=separate_space)
    words, items = evaluate(model, api, "New York")
    assert items.model_correct.isna().all()
    assert words.ranks.isna().all()
    assert words.best_rank.isna().all()
    assert words.best_layer.isna().all()
    assert not words.single_token.any()
    assert not words.in_prompt.any()


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "target,prediction,expected",
    [("21", 7, True), ("21", 8, True), ("21", 9, True),
     ("21", 12, False), ("addition", 10, True), ("addition", 11, True)],
)
def test_arithmetic_target_ranks_and_correctness_share_synonyms(
    api, target, prediction, expected,
):
    words, items = evaluate(AnswerModel(prediction), api, target, "order-ops")
    assert items.model_correct.eq(expected).all()
    assert words.single_token.all()
    for ranks in words.ranks:
        assert (ranks[-1] == 1) == expected  # Unique top logit; no tie assumption.
    if api == "batched":
        for accepted, matches in zip(
            items.target_ids, items.readout_target_match, strict=True,
        ):
            assert (prediction in accepted) == expected
            np.testing.assert_array_equal(matches, [expected, expected])
        assert words.accepted_ids.tolist() == items.target_ids.tolist()


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("target", [None, "", " ", "unknown"])
def test_unannotated_and_unsupported_targets_are_null(api, target):
    words, items = evaluate(AnswerModel(0), api, target)
    assert items.model_correct.isna().all()
    if target is None:
        assert words.empty
    else:
        assert words.ranks.isna().all()
        assert not words.single_token.any()


@pytest.mark.parametrize("api", APIS)
def test_arithmetic_expansion_is_dataset_specific(api):
    words, items = evaluate(AnswerModel(8), api, "21", "multihop")
    assert items.model_correct.eq(False).all()
    assert all(ranks[-1] > 1 for ranks in words.ranks)


def test_legacy_probe_fallback_is_preserved_but_not_used_for_answers():
    model = AnswerModel(1, separate_space=True)
    assert spelling_ids(model.tokenizer, "New York") == {1}
    assert single_token_ids(model.tokenizer, "New York") == set()
    samples = [{
        "name": "answer", "prompt": "prompt", "target": "New York",
        "intermediates": ["New York"],
    }]
    logits = model.unembed(torch.zeros(2, 1, model.d_model))
    words, item = _readout_rows(model, logits, "multihop", samples, 0, [6])
    assert words[0]["kind"] == "intermediate"
    np.testing.assert_array_equal(words[0]["ranks"], [1, 1])
    assert not words[0]["single_token"]
    assert words[1]["kind"] == "target"
    assert words[1]["ranks"] is None
    assert item["model_correct"] is None


def test_target_lookup_is_reused_not_recomputed_for_correctness():
    model = AnswerModel(8)
    samples = [{"name": "answer", "prompt": "prompt", "target": "21",
                "intermediates": []}]
    lookup = Mock(side_effect=[{8}, {7}])
    logits = model.unembed(torch.zeros(2, 1, model.d_model))
    words, item = _readout_rows(
        model, logits, "order-ops", samples, 0, [6], spelling_lookup=lookup,
    )
    lookup.assert_called_once_with("21", True)
    assert item["model_correct"] is True
    np.testing.assert_array_equal(words[0]["ranks"], [1, 1])


def test_decoded_spellings_remains_opt_in_and_preserves_all_exact_ids():
    tok = AnswerTokenizer()
    strict = DecodedSpellings(tok, vocab_size=len(tok.pieces))
    assert single_token_ids(tok, "21", expand=True) == {7, 8, 9}
    assert strict("21", True) == {7, 8, 9, 14}
    assert strict("New York") == set()
    assert strict("unknown") == set()
    assert strict("") == set()
    assert strict(" ") == set()
    assert strict("Cafe\u0301") == set()  # No Unicode normalization.
    assert strict("coffee") == set()  # No translation expansion.
    assert strict("café") == {13}
    # Duplicate decoded IDs remain opt-in; no vocabulary scan in legacy mode.
    for api, correct in [("paired", False), ("strict", True), ("batched", True)]:
        words, items = evaluate(AnswerModel(14), api, "21", "order-ops")
        assert items.model_correct.eq(correct).all()
        assert all((ranks[-1] == 1) == correct for ranks in words.ranks)
