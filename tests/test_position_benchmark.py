"""Position ranks agree with direct native readouts on small CPU models."""

from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from jlens.evaluation import identity_lens
from jlens.hooks import ActivationRecorder
from jlens.position_benchmark import (
    evaluate_position_ranks,
    latent_bridge_spans,
    load_latent_bridge,
    reduce_position_ranks,
)
from jlens.readout import readout, selected_token_ranks
from jlens.strict_scoring import ExplicitPromptModel
from tests.test_generic_evaluation import native_model  # noqa: F401


def lookup(word, expand=False):
    return {ord(word)} if len(word) == 1 else set()


def test_position_ranks_match_native_serial_and_batching(native_model):  # noqa: F811
    hf, adapter = native_model
    model = ExplicitPromptModel(adapter, bos_policy="tokenizer")
    lens = identity_lens(model)
    lens.jacobians[0][0, 1] = 0.7
    lens.jacobians[model.n_layers - 1] = 9 * torch.eye(model.d_model)
    evals = {"example": [
        {"name": "short", "prompt": "abc", "intermediates": ["a", "unknown"],
         "controls": ["d"], "target": "b"},
        {"name": "long", "prompt": "abc\ndef ", "intermediates": ["d"],
         "controls": ["a"], "target": "unknown"},
    ]}
    kwargs = dict(spelling_lookup=lookup, batch_size=8, progress=False)
    words, items = evaluate_position_ranks(model, lens, evals, **kwargs)
    assert words.attrs["evaluation"]["n_model_passes"] == 1
    unpadded, _ = evaluate_position_ranks(
        model, lens, evals, batching="equal_length", position_chunk_size=1,
        rank_chunk_size=1, **kwargs,
    )
    for left, right in zip(words.ranks, unpadded.ranks, strict=True):
        if left is None:
            assert right is None
        else:
            np.testing.assert_array_equal(left, right)
    assert items.loc[1, "model_correct"] is None
    for item in evals["example"]:
        ids = model.encode(item["prompt"])
        with torch.no_grad(), ActivationRecorder(model.layers, range(model.n_layers)) as recorder:
            expected_logits = hf(input_ids=ids, use_cache=False).logits
        actual = items[items.item.eq(item["name"])].iloc[0]
        assert actual.model_top1_id == int(expected_logits[0, -1].argmax())
        group = words[words.item.eq(item["name"])]
        assert set(group[group.kind.eq("control")].word) == set(item["controls"])
        for row in group.itertuples():
            if row.ranks is None:
                assert row.word == "unknown"
                continue
            assert all(ids[0, p] != 128 for p in row.positions)
            for layer in range(model.n_layers):
                h = recorder.activations[layer][0, row.positions].float()
                if row.lens == "J-lens" and layer < model.n_layers - 1:
                    h = lens.transport(h, layer)
                with torch.no_grad():
                    scores = readout(model, h).ranking_scores
                    expected = selected_token_ranks(scores, torch.tensor([ord(row.word)]))
                np.testing.assert_array_equal(row.ranks[layer], expected[:, 0].numpy())
        spans = {("example", item["name"]): {
            "last": [group.iloc[0].positions[-1]], "empty": [],
        }}
        reduced = reduce_position_ranks(group, spans)
        assert reduced[reduced.span.eq("empty")].ranks.isna().all()
        for before, after in zip(group.ranks, reduced[reduced.span.eq("last")].ranks, strict=True):
            if before is not None:
                np.testing.assert_array_equal(before[:, -1], after)


def test_latent_spans_use_offsets_and_split_country():
    vocab = {word: i for i, word in enumerate([
        "[UNK]", "[BOS]", "Toyota", "is", "a", "company", "from", "coun", "##try",
        "where", "the", "main", "language",
    ])}
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]",
    )
    from types import SimpleNamespace

    model = ExplicitPromptModel(SimpleNamespace(tokenizer=tokenizer), bos_policy="prepend")
    item = {"subject": "Toyota", "prompt": "Toyota is a company from a country where the main language is"}
    spans = latent_bridge_spans(model, item)
    ids = model.encode_ids(item["prompt"])
    def tokens(span):
        return tokenizer.convert_ids_to_tokens([ids[p] for p in spans[span]])
    assert tokens("subject") == ["Toyota"]
    assert tokens("slot") == ["from"]
    assert tokens("country token") == ["coun", "##try"]
    assert tokens("last") == ["is"]
    assert 0 not in spans["all"]
    assert set(spans["all but slot"]) == set(spans["all"]) - set(spans["slot"])
    with pytest.raises(ValueError, match="exceeding"):
        latent_bridge_spans(model, item, max_seq_len=3)


def test_all_candidates_and_historical_selection_are_preserved():
    evals = load_latent_bridge(Path(__file__).resolve().parents[1])
    items = [item for group in evals.values() for item in group]
    assert len(evals) == 5
    assert len(items) == 530
    assert sum(item["gpt2_kept"] for item in items) == 354
    assert all(set(item["controls"]).isdisjoint(item["intermediates"]) for item in items)
