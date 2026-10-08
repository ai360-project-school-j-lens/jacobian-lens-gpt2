"""Completed-answer scoring and real batched HF generation, without weights."""

import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import jlens
from jlens.generated_answers import (
    generate_answers_batched,
    generated_answer_counts,
    greedy_generation_config,
    score_answer,
    score_generations,
)
from jlens.strict_scoring import ExplicitPromptModel
from jlens.summaries import retrieval_summary


@pytest.mark.parametrize("text,target,dataset,expected", [
    ("14.", "14", "order-ops", True),
    ("fourteen\nExplanation", "14", "order-ops", True),
    ("14", "fourteen", "order-ops", True),
    ("**The answer is 14.**", "14", "order-ops", True),
    ("Answer: twenty-one.", "21", "order-ops", True),
    ("14.0.", "14", "order-ops", True),
    ("28/2.", "14", "order-ops", True),
    ("minus fourteen.", "-14", "order-ops", True),
    ("fourteen.", "14", "multihop", True),
    ("1.", "14", "order-ops", False),
    ("140.", "14", "order-ops", False),
    ("14.5", "14", "order-ops", False),
    ("14th", "14", "order-ops", False),
    ("1,000.", "1", "order-ops", False),
    ("15. Actually, 14.", "14", "order-ops", False),
    ("Let us calculate. 14.", "14", "order-ops", False),
    ("one two.", "12", "order-ops", False),
    ("\"Pequeño\"", "pequeño", "multilingual", True),
    ("New  York.", "new york", "multihop", True),
    ("Tirana.", "Iran", "multihop", False),
    ("Not Atlantic. Atlantic", "Atlantic", "multihop", False),
    ("cafe\u0301.", "café", "multilingual", True),
    ("Atlantic Ocean.", "Atlantic", "multihop", False),
    ("six.", "seis", "multilingual", False),
    ("\n\t", "14", "order-ops", False),
])
def test_complete_answer_scoring(text, target, dataset, expected):
    result = score_answer(text, target, dataset=dataset)
    assert result["answer_correct"] is expected


def test_aliases_and_truncated_fragments():
    assert score_answer("Atlantic Ocean.", "Atlantic", dataset="multihop",
                        aliases=["Atlantic Ocean"])["answer_correct"]
    for target, text in [("14", "14"), ("Italy", "Italy")]:
        result = score_answer(text, target, dataset="multihop", truncated=True)
        assert not result["answer_correct"]
        assert result["answer_status"] == "truncated_answer"
    assert score_answer("14. More unfinished", "14", dataset="order-ops",
                        truncated=True)["answer_correct"]
    with pytest.raises(ValueError, match="sequence"):
        score_answer("Atlantic", "Atlantic", dataset="multihop", aliases="Ocean")


class CharTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]

    def encode(self, text, *, add_special_tokens=True):
        ids = [ord(c) + 3 for c in text]
        return [0, *ids] if add_special_tokens else ids

    def decode(self, ids, *, skip_special_tokens=False, **kwargs):
        return "".join(
            chr(i - 3) if i >= 3 else ("" if skip_special_tokens else f"<special{i}>")
            for i in ids
        )

    def get_vocab(self):
        return {str(i): i for i in range(131)}


@pytest.fixture
def tiny_hf():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(3)
    hf = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=131, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        bos_token_id=0, eos_token_id=1, pad_token_id=2,
    )).eval()
    model = ExplicitPromptModel(jlens.from_hf(hf, CharTokenizer(), force_bos=False),
                                bos_policy="prepend")
    yield hf, model
    torch.set_num_threads(previous)


def sample(name, prompt, target="14"):
    return dict(name=name, prompt=prompt, target=target, intermediates=["a"])


def test_real_left_padded_generation_matches_unpadded_reference(tiny_hf):
    hf, model = tiny_hf
    evals = {"order-ops": [sample("long", "abcdef"), sample("short", "a"),
                           sample("medium", "abc")]}
    with patch.object(hf, "generate", wraps=hf.generate) as generate:
        result = generate_answers_batched(hf, model, evals, batch_size=2,
                                          max_new_tokens=4, progress=False)
        assert generate.call_count == 2
        first = generate.call_args_list[0].kwargs
        assert first["input_ids"].shape[0] == 2
        assert first["attention_mask"][0].tolist() == [0, 0, 1, 1]
        assert first["input_ids"][0, -2:].tolist() == [0, ord("a") + 3]
    assert result.item.tolist() == ["long", "short", "medium"]
    config = greedy_generation_config(hf, model.tokenizer, max_new_tokens=4)
    # Independent unpadded references expose wrong padding/continuation slicing.
    for row in result.itertuples():
        ids = model.encode(row.prompt)
        with torch.inference_mode():
            output = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 generation_config=config)
        expected = output[0, ids.shape[1]:].tolist()
        assert row.generated_ids == expected
    assert result.attrs["generation"]["batch_sizes"] == [2, 1]


def test_eos_slicing_multitoken_answers_and_unannotated_rows(tiny_hf):
    hf, model = tiny_hf
    # Actual token pieces '1', '4', '.', EOS followed by synthetic batch padding.
    def generate(**kwargs):
        ids = kwargs["input_ids"]
        tail = torch.tensor([[52, 55, 49, 1, 2]]).expand(len(ids), -1)
        return torch.cat([ids, tail], dim=1)

    evals = {"order-ops": [sample("a", "abc"), sample("b", "def", "fourteen")],
             "poetry": [dict(name="none", prompt="xyz", intermediates=["x"])]}
    with patch.object(hf, "generate", side_effect=generate) as call:
        result = generate_answers_batched(hf, model, evals, progress=False)
        assert call.call_count == 1
    assert result.generated_text.iloc[:2].tolist() == ["14.", "14."]
    assert pd.isna(result.generated_text.iloc[2])
    assert result.generated_ids.iloc[0] == [52, 55, 49, 1]
    answers = score_generations(result)
    assert answers.answer_correct.tolist() == [True, True, None]
    counts = generated_answer_counts(answers).set_index("dataset")
    assert counts.loc["order-ops", "generated_accuracy"] == 1
    assert counts.loc["poetry", "unannotated"] == 1
    assert pd.isna(counts.loc["poetry", "generated_accuracy"])


def test_generation_rejects_truncation_and_duplicate_items(tiny_hf):
    hf, model = tiny_hf
    with patch.object(hf, "generate") as generate:
        with pytest.raises(ValueError, match="exceeding"):
            generate_answers_batched(hf, model, {"d": [sample("a", "abcdefgh")]},
                                     max_seq_len=3)
        with pytest.raises(ValueError, match="max_batch_tokens"):
            generate_answers_batched(hf, model, {"d": [sample("a", "abc")]},
                                     max_batch_tokens=4)
        with pytest.raises(ValueError, match="Duplicate"):
            generate_answers_batched(hf, model, {"d": [sample("a", "abc")] * 2})
        generate.assert_not_called()


def test_generated_correct_subset_does_not_require_single_token_target():
    items = pd.DataFrame([
        dict(dataset="d", item=item, target="14", model_correct=token_correct,
             target_ids=ids, answer_correct=answer_correct, lens=lens)
        for item, token_correct, ids, answer_correct in [
            ("multi", None, (), True), ("token_only", True, (3,), False),
        ] for lens in ["logit lens", "J-lens"]
    ])
    words = pd.DataFrame([
        dict(dataset="d", item=item, lens=lens, kind="intermediate", word="a",
             role=0, single_token=True, ranks=ranks)
        for item, ranks in [("multi", [1, 10]), ("token_only", [10, 1])]
        for lens in ["logit lens", "J-lens"]
    ])
    before = items.copy(deep=True)
    old = retrieval_summary(words, items, k=1, subsets=("correct",))
    new = retrieval_summary(words, items, k=1, subsets=("correct",),
                            correctness_column="answer_correct")
    assert old.score.tolist() == [0, 0]
    assert new.score.tolist() == [1, 1]
    assert new.n_items_used.tolist() == [1, 1]
    pd.testing.assert_frame_equal(items, before)
    items.loc[1, "answer_correct"] = False
    with pytest.raises(ValueError, match="Inconsistent lens copies"):
        retrieval_summary(words, items, correctness_column="answer_correct")


def test_new_notebook_cells_run_offline_and_cache_generation(tiny_hf, tmp_path):
    from jlens.batched_evaluation import evaluate_paired_batched
    from jlens.evaluation import identity_lens
    from jlens.generated_answers import SCORING_VERSION
    from jlens.strict_scoring import DecodedSpellings
    from jlens.summaries import (
        answer_counts,
        plot_retrieval_summary,
        target_final_counts,
    )

    hf, model = tiny_hf
    path = (Path(__file__).resolve().parents[1]
            / "notebooks/jacobian_lens/pretrained_generated_answer_summary.ipynb")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
    datasets = ["multihop", "multilingual", "order-ops", "poetry"]
    def load_eval(_, name):
        return [sample("first", "a "), sample("second", "abc ")]

    cache = {}
    def fingerprint(value):
        return json.dumps(value, sort_keys=True)

    def cached(namespace, settings, compute, **kwargs):
        key = (namespace, fingerprint(settings))
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    ns = dict(
        pd=pd, np=np, torch=torch, plt=plt, model=model, hf_model=hf, tokenizer=model.tokenizer,
        hashlib=hashlib, inspect=inspect,
        fitted_lens=identity_lens(model),
        spellings=DecodedSpellings(model.tokenizer, vocab_size=131),
        DATASETS=datasets, REPO_DIR=tmp_path, K=10, load_eval=load_eval,
        TASK_PROMPT_POLICY="trim_completion_spaces", BATCH_OPTIONS=dict(batch_size=4, progress=False),
        GENERATION_OPTIONS=dict(batch_size=4, max_new_tokens=4, progress=False),
        REFRESH_TASKS=False, REFRESH_GENERATIONS=False, ANSWER_ALIASES={},
        require_run=lambda: None, implementation_hashes=lambda *names: dict.fromkeys(names, "test"),
        fingerprint=fingerprint, cached=cached, display=lambda *args: None,
        evaluate_paired_batched=evaluate_paired_batched,
        generate_answers_batched=generate_answers_batched,
        greedy_generation_config=greedy_generation_config,
        score_generations=score_generations, generated_answer_counts=generated_answer_counts,
        SCORING_VERSION=SCORING_VERSION, answer_counts=answer_counts,
        retrieval_summary=retrieval_summary, plot_retrieval_summary=plot_retrieval_summary,
        target_final_counts=target_final_counts,
    )
    # Run task, token summary, generation, scoring and both retrieval sections.
    sources = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    selected = [s for s in sources if s.startswith((
        "all_words =", "require_task_results()", "require_answers()",
    ))]
    with patch.object(plt, "show"), patch.object(hf, "generate", wraps=hf.generate) as generate:
        for source in selected:
            if "intermediate_rank_sweep" not in source:
                exec(source, ns)
        count = generate.call_count
        assert count == 2
        generation_cell = next(s for s in selected if "generations = cached(" in s)
        score_cell = next(s for s in selected if "answers = score_generations(" in s)
        exec(generation_cell, ns)
        exec(score_cell, ns)
        assert generate.call_count == count  # Persistent-cache contract, no new inference.
        ns["ANSWER_ALIASES"] = {"multihop": {"14": ["fourteen"]}}
        with pytest.raises(RuntimeError, match="stale"):
            ns["require_answers"]()
        exec(score_cell, ns)
        assert generate.call_count == count
        ns["GENERATION_OPTIONS"]["max_new_tokens"] = 5
        with pytest.raises(RuntimeError, match="stale"):
            exec(score_cell, ns)
    assert "next_token_target_match" in ns["items"]
    plt.close("all")
