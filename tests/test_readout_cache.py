"""The readout cache reproduces evaluate_paired ranks without re-running the model."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import jlens
from jlens.evaluation import evaluate_paired, pass_at_k, single_token_ids
from jlens.lens import JacobianLens
from jlens.readout_cache import (
    READOUT_CACHE_VERSION,
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


def test_cached_targets_match_direct_complete_answer_policy(tmp_path):
    model, lens = _model_and_lens()
    answers = ["q", "division", "long", "New York", "", " ", "\t", "<BOS>"]
    evals = {"order-ops": [
        {"name": str(i), "prompt": "abc def", "intermediates": ["long"],
         "target": answer}
        for i, answer in enumerate(answers)
    ]}
    cache = build_readout_cache(model, evals)
    cache.save(tmp_path / "cache.pt")
    cache = ReadoutCache.load(tmp_path / "cache.pt")
    targets = cache.words[cache.words.kind == "target"]
    assert targets.ids.tolist() == [
        sorted(single_token_ids(model.tokenizer, answer, expand=True))
        for answer in answers
    ]
    assert all(not ids for ids in targets.ids.tolist()[2:])
    assert ord("/") in targets.ids.iloc[1]  # Arithmetic synonyms remain accepted.
    probes = cache.words[cache.words.kind == "intermediate"]
    assert probes.ids.tolist() == [[ord(" ")]] * len(answers)

    ranker = WordRanker(cache.words)
    ranks = ranker.rank_residuals(model.unembed, cache.H.transpose(0, 1)).T
    frame = words_frame(cache.words, ranks)
    direct, _ = evaluate_paired(model, lens, evals)
    direct = direct[direct.lens == "logit lens"].reset_index(drop=True)
    for cached, reference in zip(frame.itertuples(), direct.itertuples(), strict=True):
        assert cached.single_token == reference.single_token
        assert cached.in_prompt == reference.in_prompt
        if reference.ranks is None:
            assert cached.ranks is None
            assert np.isnan(cached.best_rank)
            assert np.isnan(cached.best_layer)
            assert np.isnan(cached.inner_best)
        else:
            np.testing.assert_array_equal(cached.ranks, reference.ranks)

    scores = item_scores(cache.words, ranks, "order-ops", "target", k=129)
    np.testing.assert_array_equal(scores[:2], [1, 1])
    assert np.isnan(scores[2:]).all()
    summary = pass_at_k(frame, ks=[129])
    target = summary[summary.kind == "target"].iloc[0]
    assert target.score == 1
    assert target.n_items == len(answers)
    assert target.n_items_scored == 2


@pytest.mark.parametrize("ids", [[[], [2, 4], []], [[], []], []])
@pytest.mark.parametrize("row_chunk", [1, 3])
def test_ranker_preserves_empty_accepted_sets(ids, row_chunk):
    words = pd.DataFrame({"item_index": [0] * len(ids), "ids": ids})
    logits = torch.zeros(2, 3, 1, 8)  # Multiple leading dimensions, exact ties.
    logits[1] = -torch.inf  # Padding must not win ties at negative infinity.
    ranks = WordRanker(words)(logits, row_chunk=row_chunk)
    assert ranks.shape == (2, 3, len(ids))
    for i, accepted in enumerate(ids):
        if accepted:
            np.testing.assert_array_equal(ranks[..., i], min(accepted) + 1)
        else:
            assert np.isnan(ranks[..., i]).all()


@pytest.mark.parametrize("k", [1, None])
@pytest.mark.parametrize("all_unsupported", [False, True])
def test_cached_scores_exclude_unsupported_without_nan_contamination(k, all_unsupported):
    words = pd.DataFrame({
        "dataset": ["d"] * 4, "kind": ["target"] * 4,
        "item_index": [0, 0, 1, 2], "item": ["a", "a", "b", "c"],
        "ids": [[], [], [], []] if all_unsupported else [[1], [], [], [2]],
    })
    # Include stale finite ranks for empty IDs: acceptance, not ranks, is decisive.
    ranks = np.array([[1, 2], [1, 1], [np.nan, np.nan], [10, 1]])
    scores = item_scores(words, np.stack([ranks, ranks]), "d", "target", k=k)
    expected = [np.nan] * 3 if all_unsupported else (
        [1, np.nan, 0] if k else [0, np.nan, 1]
    )
    np.testing.assert_array_equal(scores, [expected, expected])
    frame = words_frame(words, ranks)
    for row in frame.loc[words.ids.map(len) == 0].itertuples():
        assert row.ranks is None
        assert np.isnan(row.best_rank)
        assert np.isnan(row.best_layer)
        assert np.isnan(row.inner_best)
    summary = pass_at_k(frame, ks=[1])
    if all_unsupported:
        assert summary.score.isna().all()
        assert (summary.n_items_scored == 0).all()


@pytest.mark.parametrize("version", [None, 1, READOUT_CACHE_VERSION + 1])
def test_readout_cache_rejects_incompatible_scoring(tmp_path, version):
    path = tmp_path / "legacy.pt"
    raw = {"H": torch.zeros(1, 2, 3), "items": {"target": ["long"]},
           "words": {"kind": ["target"], "ids": [[32]]}}
    if version is not None:
        raw["version"] = version
    torch.save(raw, path)
    with pytest.raises(ValueError, match="rebuild the cache.*derived ranks/results"):
        ReadoutCache.load(path)


def test_readout_cache_records_scoring_version(tmp_path):
    path = tmp_path / "current.pt"
    cache = ReadoutCache(
        torch.zeros(1, 2, 3), pd.DataFrame({"target": ["long"]}),
        pd.DataFrame({"kind": ["target"], "ids": [[]]}),
    )
    cache.save(path)
    assert torch.load(path, weights_only=False)["version"] == READOUT_CACHE_VERSION
    loaded = ReadoutCache.load(path)
    assert loaded.words.ids.tolist() == [[]]
    torch.testing.assert_close(cache.H, loaded.H)


@pytest.mark.parametrize("notebook, artifacts", [
    ("fit_convergence_ci", ["eval_cache", "baseline_words", "ranks_lenses", "lenses_meta"]),
    ("residual_deltas", ["eval_cache", "deltas_ranks"]),
])
def test_notebook_derived_caches_use_scoring_version(notebook, artifacts):
    path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "jacobian_lens"
        / f"{notebook}.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    filenames = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "READOUT_CACHE_VERSION" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            # Only versioned artifact names; unrelated display f-strings need state.
            if any(
                isinstance(value, ast.Constant)
                and any(str(value.value).startswith(name + "_v") for name in artifacts)
                for value in node.values
            ):
                expression = compile(ast.Expression(node), str(path), "eval")
                filenames.append(eval(expression, {"READOUT_CACHE_VERSION": 2}))
                assert "v3." in eval(expression, {"READOUT_CACHE_VERSION": 3})
    assert {name.rsplit("_v", 1)[0] for name in filenames} == set(artifacts)
