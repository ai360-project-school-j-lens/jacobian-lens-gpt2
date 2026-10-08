import numpy as np
import pandas as pd
import pytest
import torch

from jlens.generation_readout import (
    Continuation,
    continuation_ranks,
    cut_continuation,
    future_words,
    hit_rates,
    item_bootstrap_auc_diff,
    step_texts,
    word_pattern,
    word_status,
)
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder

VOCAB = ["<eos>", " The", " planet", " Mars", " is", " red", ".", "\n", " Ma", "rs", " 5"]


class WordTokenizer:
    """Decodes IDs by concatenating fixed strings; enough for the text helpers."""

    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=False, **_kw):
        return "".join(
            VOCAB[i] for i in ids if not (skip_special_tokens and i == 0)
        )


def test_word_pattern_matches_whole_words_and_symbols():
    pattern = word_pattern(["Mars"])
    assert pattern.search("the planet mars is")
    assert not pattern.search("Marsh land")
    star = word_pattern(["multiplication", "*", "times"])
    assert star.search("(2 + 3) * 4")
    assert not word_pattern(["5"]).search("25 apples")


def test_word_status_orders_in_text_next_latent():
    pattern = word_pattern(["Mars"])
    assert word_status("The Mars", " is red", pattern) == "in_text"
    assert word_status("The planet", " Mars is", pattern) == "next"
    assert word_status("The", " planet Mars", pattern) == "latent"


def test_step_texts_and_statuses_follow_steps():
    tok = WordTokenizer()
    cont = Continuation([1, 2], [8, 9, 4, 5], "max_new_tokens")  # " Ma"+"rs" = Mars
    contexts, rests = step_texts(tok, cont)
    assert contexts[0] == " The planet" and rests[0] == " Mars is red"
    pattern = word_pattern(["Mars"])
    statuses = [word_status(c, r, pattern) for c, r in zip(contexts, rests, strict=True)]
    # Multi-token word: "next" at the step that starts it, then mid-word, then read.
    assert statuses == ["next", "latent", "in_text", "in_text"]
    assert cont.start == 1 and cont.n_steps == 4 and cont.token_ids == [1, 2, 8, 9, 4, 5]


def test_cut_continuation_stops_at_eos_and_newline():
    tok = WordTokenizer()
    assert cut_continuation(tok, [3, 0, 4], {0}, stop_at_newline=True) == ([3, 0], "eos")
    assert cut_continuation(tok, [7, 3, 7, 4], {0}, stop_at_newline=True) == (
        [7, 3, 7], "newline"
    )
    assert cut_continuation(tok, [3, 7, 4], {0}, stop_at_newline=False) == (
        [3, 7, 4], "max_new_tokens"
    )


def test_future_words_skip_prompt_words_and_stopwords():
    tok = WordTokenizer()
    cont = Continuation([1, 2], [4, 8, 9, 4, 5, 6], "max_new_tokens")
    words = future_words(tok, cont, min_letters=3)
    # " is" is a stopword, "planet" is in the prompt; Mars starts at step 1.
    assert words == [("Mars", 1), ("red", 4)]


def _identity(model):
    eye = torch.eye(model.d_model)
    return JacobianLens(
        {layer: eye for layer in range(model.n_layers - 1)},
        n_prompts=1, d_model=model.d_model,
    )


def test_continuation_ranks_match_direct_ranking():
    model = TinyDecoder(n_layers=3, d_model=8, vocab_size=32)
    cont = Continuation([0, 5, 6, 7], [8, 9, 10], "max_new_tokens")
    id_sets = [[3], [4, 11], []]
    ranks = continuation_ranks(model, _identity(model), cont, id_sets)
    assert ranks.shape == (2, 3, 3, 3)
    np.testing.assert_array_equal(ranks[0], ranks[1])  # J = I reduces to the logit lens
    assert (ranks[..., 2] == 0).all()  # unsupported word
    hidden = model.embed_tokens(torch.tensor([cont.token_ids]))
    for block in model.layers:
        hidden = block(hidden)
    logits = model.unembed(hidden[0, cont.start:cont.start + cont.n_steps])
    order = torch.sort(-logits, dim=-1, stable=True).indices.argsort(-1) + 1
    expected = torch.minimum(order[:, 4], order[:, 11]).numpy()
    np.testing.assert_array_equal(ranks[0, :, -1, 1], expected)


def test_continuation_ranks_transport_changes_inner_layers_only():
    model = TinyDecoder(n_layers=3, d_model=8, vocab_size=32)
    torch.manual_seed(1)
    lens = JacobianLens(
        {layer: torch.randn(8, 8) for layer in range(2)}, n_prompts=1, d_model=8,
    )
    cont = Continuation([0, 5, 6], [8, 9], "max_new_tokens")
    ranks = continuation_ranks(model, lens, cont, [list(range(1, 32))[:5]])
    np.testing.assert_array_equal(ranks[0, :, -1], ranks[1, :, -1])
    assert not np.array_equal(ranks[0, :, :2], ranks[1, :, :2])


def test_hit_rates_weight_items_equally_and_auc_bounds():
    frame = pd.DataFrame({
        "item": ["a", "a", "a", "b"],
        "best_rank": [1, 1, 1000, 1000],
        "group": "g",
    })
    table = hit_rates(frame, ["group"], ks=(1, 10))
    assert table.loc[0, "pass@1"] == pytest.approx((2 / 3 + 0) / 2)
    perfect = hit_rates(frame.assign(best_rank=1), ["group"], ks=(1, 10))
    assert perfect.loc[0, "auc"] == pytest.approx(1.0)


def test_bootstrap_auc_diff_is_paired():
    frame = pd.DataFrame({
        "item": ["a", "b", "a", "b"],
        "lens": ["J-lens", "J-lens", "logit lens", "logit lens"],
        "best_rank": [1, 1, 100, 100],
    })
    diff, lo, hi = item_bootstrap_auc_diff(frame, ks=(1, 100), n_boot=50)
    assert diff == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
