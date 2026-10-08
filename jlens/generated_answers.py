"""Batched raw-prompt completions and conservative, complete-answer scoring.

Generation is independent of lens readouts. Cache its returned table before
scoring so aliases and extraction rules can change without another model pass.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from fractions import Fraction

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig

from jlens.evaluation import synonyms
from jlens.strict_scoring import ExplicitPromptModel

SCORING_VERSION = "first-complete-phrase-v1"
_KEYS = ["dataset", "item"]
_GENERATION_COLUMNS = [
    *_KEYS, "prompt", "target", "generated_text", "generated_text_raw",
    "generated_ids", "ended_with_eos", "generation_truncated",
]
# A decimal point inside a number is not a sentence boundary. Apostrophes
# inside words remain intact. This deliberately does not search later answers.
_BOUNDARY = re.compile(r"[\n\r;:!?。！？；，\"”»]|,(?!\d)|(?<!\w)'|\.(?!\d)")
_PREFIX = re.compile(r"^(?:the answer is|answer is|answer\s*:|=)\s*", re.I)
_NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)\Z")
_NUMBER_WORDS = {
    synonyms(str(n))[1].replace("-", " "): n for n in range(100)
}


def normalize_answer(text: str) -> str:
    """Normalize Unicode, case and whitespace without substring matching."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _number(text: str) -> Fraction | None:
    text = normalize_answer(text).replace("−", "-")
    if _NUMERIC.fullmatch(text):
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError):
            return None
    words = text.replace("-", " ")
    sign = 1
    if words.startswith(("minus ", "negative ")):
        sign, words = -1, words.split(" ", 1)[1]
    value = _NUMBER_WORDS.get(words)
    return None if value is None else Fraction(sign * value)


def score_answer(
    continuation: str,
    target: str,
    *,
    dataset: str,
    truncated: bool = False,
    aliases: Sequence[str] = (),
) -> dict:
    """Compare the first completed phrase, never a fragment or later mention.

    Strip leading whitespace/quotes/backticks/asterisks and an optional English
    answer prefix. The first sentence/line/quote/clause boundary ends the answer.
    Without a boundary, the end of an EOS-terminated continuation ends it;
    a token-limit-truncated phrase is rejected. Numeric targets and order-ops
    accept equivalent digits, decimals, fractions, and English integers 0..99.
    Other answers require exact normalized equality or an explicit alias.
    Unparsed/truncated answers count as misses, with reasons for manual review.
    """
    if isinstance(aliases, str) or any(not isinstance(a, str) for a in aliases):
        raise ValueError("aliases must be a sequence of complete answer strings")
    text = continuation.lstrip(" \t\r\n\"'“‘«`*")
    text = _PREFIX.sub("", text, count=1).lstrip(" \t\"'“‘«`*")
    boundary = _BOUNDARY.search(text)
    answer = (text[:boundary.start()] if boundary else text).strip().rstrip("`*’'")
    normalized = normalize_answer(answer)
    if not normalized:
        status, correct = "empty_answer", False
    elif truncated and boundary is None:
        status, correct = "truncated_answer", False
    else:
        accepted = [target, *aliases]
        numeric = dataset == "order-ops" or bool(_NUMERIC.fullmatch(target.strip()))
        if numeric:
            value = _number(answer)
            values = {_number(word) for word in accepted} - {None}
            correct = value is not None and value in values
            status = "match" if correct else (
                "wrong_answer" if value is not None else "unparsed_numeric_answer"
            )
        else:
            correct = normalized in {normalize_answer(word) for word in accepted}
            status = "match" if correct else "wrong_answer"
    return {
        "generated_answer": answer,
        "normalized_answer": normalized,
        "answer_correct": correct,
        "answer_status": status,
    }


def greedy_generation_config(hf_model, tokenizer, *, max_new_tokens: int) -> GenerationConfig:
    """Explicit greedy decoding; retain only the checkpoint's stop/pad IDs."""
    if isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    native = hf_model.generation_config
    eos = native.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, (list, tuple)) and eos else eos
    if pad is None:
        raise ValueError("Generation needs a tokenizer pad or EOS token")
    return GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        num_return_sequences=1, use_cache=True, repetition_penalty=1.0,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=eos, pad_token_id=pad,
    )


@torch.inference_mode()
def generate_answers_batched(
    hf_model,
    model: ExplicitPromptModel,
    evals: Mapping[str, Sequence[dict]],
    *,
    batch_size: int = 8,
    max_batch_tokens: int = 2048,
    max_seq_len: int = 512,
    max_new_tokens: int = 32,
    progress: bool = True,
) -> pd.DataFrame:
    """Generate once per annotated item using the lens model's exact encoding.

    Sort by length and LEFT-pad for HF decoder-only generate(), which samples
    at the last column. Attention masks keep pads out of the context. This does
    not mutate tokenizer padding/BOS settings. Lens evaluation can independently
    right-pad and gather each row's last real position. The token budget includes
    the maximum continuation to bound KV-cache growth. Empty/unannotated targets
    get null generation fields. No per-token GPU-to-CPU synchronization is added.
    """
    if not isinstance(model, ExplicitPromptModel):
        raise TypeError("Supply ExplicitPromptModel with an explicit BOS policy")
    if hf_model.training or getattr(hf_model.config, "is_encoder_decoder", False):
        raise ValueError("Generation requires an eval-mode decoder-only model")
    for name, value in (("batch_size", batch_size), ("max_batch_tokens", max_batch_tokens),
                        ("max_seq_len", max_seq_len), ("max_new_tokens", max_new_tokens)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    config = greedy_generation_config(hf_model, model.tokenizer, max_new_tokens=max_new_tokens)
    rows, pending, seen = [], [], set()
    for dataset, samples in evals.items():
        for item in samples:
            key = (dataset, item["name"])
            if key in seen:
                raise ValueError(f"Duplicate dataset/item: {key}")
            seen.add(key)
            target = item.get("target")
            row = dict.fromkeys(_GENERATION_COLUMNS)
            row.update(dataset=dataset, item=item["name"], prompt=item["prompt"], target=target)
            rows.append(row)
            if target is None:
                continue
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"Invalid annotated target: {key}")
            ids = model.encode_ids(item["prompt"], max_length=max_seq_len)
            if len(ids) + max_new_tokens > max_batch_tokens:
                raise ValueError(f"Prompt plus continuation exceeds max_batch_tokens: {key}")
            pending.append((len(rows) - 1, ids))
    batches, batch = [], []
    for entry in sorted(pending, key=lambda entry: len(entry[1])):
        if batch and (len(batch) == batch_size or
                      (len(batch) + 1) * (len(entry[1]) + max_new_tokens) > max_batch_tokens):
            batches.append(batch)
            batch = []
        batch.append(entry)
    if batch:
        batches.append(batch)
    eos = config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else ([] if eos is None else [eos]))
    for batch in tqdm(batches, desc="greedy answer batches", disable=not progress):
        width = max(len(ids) for _, ids in batch)
        input_ids = torch.full((len(batch), width), config.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for row, (_, ids) in enumerate(batch):
            input_ids[row, -len(ids):] = torch.tensor(ids)
            mask[row, -len(ids):] = 1
        output = hf_model.generate(
            input_ids=input_ids.to(model.input_device),
            attention_mask=mask.to(model.input_device), generation_config=config,
        )
        continuations = output[:, width:].cpu().tolist()
        for (index, _), ids in zip(batch, continuations, strict=True):
            end = next((i for i, token in enumerate(ids) if token in eos_ids), None)
            ended = end is not None
            ids = ids if end is None else ids[:end + 1]
            rows[index].update(
                generated_text=model.tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                ),
                generated_text_raw=model.tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
                ),
                generated_ids=ids, ended_with_eos=ended,
                generation_truncated=not ended,
            )
        del output
    result = pd.DataFrame(rows, columns=_GENERATION_COLUMNS)
    result.attrs["generation"] = {
        "n_items": len(rows), "n_generated": len(pending), "n_batches": len(batches),
        "batch_sizes": [len(batch) for batch in batches], "padding_side": "left",
        "bos_policy": model.bos_policy, "generation_config": config.to_dict(),
        "max_batch_tokens": max_batch_tokens, "max_seq_len": max_seq_len,
    }
    return result


def score_generations(
    generations: pd.DataFrame,
    *,
    aliases: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> pd.DataFrame:
    """Score cached completions; aliases are keyed by dataset then exact target."""
    if generations.duplicated(_KEYS).any():
        raise ValueError("Expected one generation per dataset/item")
    rows = []
    for row in generations.to_dict("records"):
        if pd.isna(row["target"]):
            score = dict(generated_answer=None, normalized_answer=None,
                         answer_correct=None, answer_status="unannotated")
        else:
            if not isinstance(row["generated_text"], str):
                raise ValueError("Missing generation for an annotated target")
            score = score_answer(
                row["generated_text"], row["target"], dataset=row["dataset"],
                truncated=bool(row["generation_truncated"]),
                aliases=(aliases or {}).get(row["dataset"], {}).get(row["target"], ()),
            )
        rows.append({**row, **score})
    return pd.DataFrame(rows, columns=[
        *generations.columns, "generated_answer", "normalized_answer",
        "answer_correct", "answer_status",
    ])


def generated_answer_counts(answers: pd.DataFrame) -> pd.DataFrame:
    """Accuracy over ALL annotated items, including multi-token-only targets."""
    if answers.duplicated(_KEYS).any():
        raise ValueError("Count each dataset/item once")
    rows = []
    for dataset, group in answers.groupby("dataset", sort=False):
        annotated = group.target.notna()
        if group.loc[annotated, "answer_correct"].isna().any():
            raise ValueError("All annotated answers must be scored")
        n = int(annotated.sum())
        correct = int(group.answer_correct.eq(True).sum())
        rows.append(dict(
            dataset=dataset, total=len(group), annotated=n, unannotated=len(group) - n,
            correct=correct, incorrect=n - correct,
            generated_accuracy=correct / n if n else float("nan"),
            truncated=int(group.generation_truncated.eq(True).sum()),
        ))
    return pd.DataFrame(rows)
