"""The readout cache reproduces evaluate_paired ranks without re-running the model."""

from types import SimpleNamespace

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import jlens
from jlens.evaluation import evaluate_paired, pass_at_k
from jlens.lens import JacobianLens
from jlens.readout_cache import (
    ReadoutCache,
    WordRanker,
    build_readout_cache,
    item_scores,
    words_frame,
)


class _Tokenizer:
    all_special_ids = [128]
    bos_token_id = 128

    def encode(self, text, *, add_special_tokens=True):
        ids = [ord(char) % 128 for char in text]
        return [128, *ids] if add_special_tokens else ids

    def __call__(self, text, *, max_length=512, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)[:min(max_length, 128)]]))

    def decode(self, ids, **kwargs):
        return "".join(chr(token) if token < 128 else "<BOS>" for token in ids)


EVALS = {
    "multihop": [
        {"name": "a", "prompt": "abc def gh", "intermediates": ["d", "x"], "target": "q"},
        {"name": "b", "prompt": "the quick fox", "intermediates": ["k"], "target": "r"},
        {"name": "c", "prompt": "zz yy xx", "intermediates": ["y", "long"], "target": "s"},
    ],
    "poetry": [
        {"name": "p", "prompt": "roses red\nviolets", "intermediates": ["v"]},
        {"name": "q", "prompt": "sky blue\nsea", "intermediates": ["b"]},
    ],
}


def _model_and_lens():
    torch.manual_seed(0)
    hf = GPT2LMHeadModel(GPT2Config(vocab_size=129, n_positions=128, n_embd=16, n_layer=3, n_head=2))
    model = jlens.from_hf(hf, _Tokenizer())
    jacobians = {layer: torch.eye(16) + 0.3 * torch.randn(16, 16) for layer in range(model.n_layers - 1)}
    return model, JacobianLens(jacobians=jacobians, n_prompts=1, d_model=16)


def test_cached_ranks_match_evaluate_paired(tmp_path):
    model, lens = _model_and_lens()
    cache = build_readout_cache(model, EVALS)
    cache.save(tmp_path / "cache.pt")
    cache = ReadoutCache.load(tmp_path / "cache.pt")
    reference, _ = evaluate_paired(model, lens, EVALS)
    ranker = WordRanker(cache.words)
    n_inner = model.n_layers - 1
    final = ranker(model.unembed(cache.H[:, -1]))
    transported = torch.stack([cache.H[:, layer] @ lens.jacobians[layer].T for layer in range(n_inner)])
    readouts = {
        "J-lens": transported,
        "logit lens": cache.H[:, :n_inner].transpose(0, 1),
    }
    for name, residuals in readouts.items():
        inner = ranker.rank_residuals(model.unembed, residuals, batch=2)
        ranks = np.concatenate([inner.T, final[:, None]], axis=1)
        expected = reference[reference.lens == name].reset_index(drop=True)
        frame = words_frame(cache.words, ranks)
        for column in ("dataset", "item", "kind", "word", "role", "single_token", "in_prompt", "best_rank"):
            assert (frame[column].values == expected[column].values).all(), (name, column)
        np.testing.assert_array_equal(np.stack(frame.ranks), np.stack(expected.ranks))

        scores = pass_at_k(frame, ks=[1, 5], layers=slice(None, -1)).set_index(["dataset", "kind", "k"]).score
        for (dataset, kind, k), score in scores.items():
            per_item = item_scores(cache.words, ranks, dataset, kind, k=k)
            assert np.isclose(np.nanmean(per_item), score), (name, dataset, kind, k)
        # Batched over leading axes.
        batched = item_scores(cache.words, np.stack([ranks, ranks]), "multihop", "intermediate", k=5)
        np.testing.assert_array_equal(batched[0], batched[1])
