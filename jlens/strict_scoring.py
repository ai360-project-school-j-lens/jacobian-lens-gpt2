"""Opt-in whole-token scoring and explicit prompt encoding for new notebooks.

The legacy evaluator's defaults are deliberately unchanged. Gloss translations
are presentation only: accepted spellings come from exact tokenizer decoding,
not vocabulary marker strings, re-encoding a displayed token, or translations.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import torch

from jlens.evaluation import synonyms


class DecodedSpellings:
    """Index complete non-special decoded tokens; never use prefix fallbacks.

    Accept as-is/lower/capitalized spellings, with zero or one leading ASCII
    space, and optionally the evaluator's order-ops synonyms. Matching is exact:
    no stripping, Unicode normalization, whitespace-only tokens, or glossary
    expansion. Multiple token IDs with the same decoded spelling are retained.
    This measures lexical token availability, not contextual segmentation or
    the probability/accuracy of a generated multi-token answer.
    """

    def __init__(self, tokenizer, *, vocab_size: int) -> None:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        special_ids.update(
            token for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
            if (token := getattr(tokenizer, attr, None)) is not None
        )
        self.by_text: dict[str, set[int]] = defaultdict(set)
        # get_vocab supplies actual IDs; the output head may include padding IDs.
        for token_id in set(tokenizer.get_vocab().values()):
            if token_id in special_ids or not 0 <= token_id < vocab_size:
                continue
            text = tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False,
                skip_special_tokens=False,
            )
            if text and not text.isspace() and "\ufffd" not in text:
                self.by_text[text].add(token_id)

    @lru_cache(maxsize=None)
    def __call__(self, word: str, expand: bool = False) -> frozenset[int]:
        ids = set()
        for spelling in synonyms(word) if expand else [word]:
            if not spelling or spelling.isspace():
                continue
            for variant in {spelling, spelling.lower(), spelling.capitalize()}:
                for text in (variant, " " + variant):
                    ids.update(self.by_text.get(text, ()))
        return frozenset(ids)


class ExplicitPromptModel:
    """Delegate a LensModel but make BOS and length handling explicit everywhere.

    ``none`` uses raw text without automatic specials (Qwen base default here).
    ``prepend`` inserts exactly one BOS before raw text; ``tokenizer`` retains
    the checkpoint tokenizer's special-token policy. Never strips whitespace or
    silently truncates the readout away. No chat template is applied.
    """

    def __init__(self, model, *, bos_policy: str = "none") -> None:
        if bos_policy not in {"none", "prepend", "tokenizer"}:
            raise ValueError("bos_policy must be none, prepend, or tokenizer")
        if bos_policy == "prepend" and model.tokenizer.bos_token_id is None:
            raise ValueError("prepend requires a tokenizer BOS token ID")
        self.model = model
        self.bos_policy = bos_policy

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode_ids(self, text: str, *, max_length: int = 512) -> list[int]:
        """Encode on CPU for length grouping without per-prompt device transfers."""
        ids = self.tokenizer.encode(
            text, add_special_tokens=self.bos_policy == "tokenizer",
        )
        if self.bos_policy == "prepend":
            ids = [self.tokenizer.bos_token_id, *ids]
        if not ids:
            raise ValueError("The prompt has no tokens")
        if len(ids) > max_length:
            raise ValueError(
                f"Prompt has {len(ids)} tokens, exceeding {max_length}; "
                "shorten it explicitly rather than silently moving the readout"
            )
        return list(ids)

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        ids = self.encode_ids(text, max_length=max_length)
        return torch.tensor([ids], dtype=torch.long, device=self.input_device)
