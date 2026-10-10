"""Cross-lingual whole-token scoring through a token glossary (opt-in protocol).

``DecodedSpellings`` accepts only tokens that *spell* an annotated word, so a
multilingual model that reads a concept out in another language (Qwen's
``意大利`` for *Italy*, ``二十`` for *20*) scores a miss. ``TranslatedSpellings``
also accepts tokens that *mean* the word, using a token-ID -> English glossary
such as ``assets/qwen_gloss.json.gz``:

* strict spellings, exactly as ``DecodedSpellings`` (always a subset);
* tokens whose surface text equals a form of the word after NFKC folding,
  casefolding and whitespace stripping (full-width ``２０`` for ``20``);
  forms are the word, its order-ops synonyms when ``expand`` is set, and its
  explicit ``aliases`` (e.g. ``乘以`` for *multiplication*);
* tokens whose gloss equals the word, its number word (``二十`` -> *twenty*
  for ``20``) or an alias. Operation synonyms such as *product* or *times* are
  deliberately not translated: their other senses (``产品``, ``次数``) are not
  the operation;
* for a word written in another language, tokens sharing the gloss of the
  word's own strict tokens (target ``黄`` -> gloss *yellow* -> `` yellow``,
  ``黄色``).

Still whole tokens only: no prefixes and no multi-token words. The glossary is
third-party display metadata of unverified provenance; results under this
protocol must be reported separately from strict results, with the matched
tokens shown for inspection.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping

from jlens.evaluation import synonyms
from jlens.strict_scoring import DecodedSpellings

__all__ = ["TranslatedSpellings", "meaning_key"]

_GLOSS_PUNCTUATION = " \t\n\"'“”‘’「」『』()（）[]【】.,。，、!！?？:：;；"


def meaning_key(text: str) -> str:
    """NFKC-folded, casefolded text with whitespace runs collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class TranslatedSpellings:
    """Token IDs that spell or mean a word, via surface folding and a glossary.

    ``glossary`` maps token IDs to English glosses; glosses listed in
    ``ignore_glosses`` (e.g. ``"fragment"``) are placeholders and never match.
    Special and out-of-head token IDs are never accepted. ``source`` tells
    whether an accepted ID is a strict spelling or a translation.
    """

    def __init__(
        self,
        tokenizer,
        *,
        vocab_size: int,
        glossary: Mapping[int, str],
        aliases: Mapping[str, Iterable[str]] | None = None,
        ignore_glosses: Iterable[str] = ("fragment",),
    ) -> None:
        self.strict = DecodedSpellings(tokenizer, vocab_size=vocab_size)
        ignored = {meaning_key(g) for g in ignore_glosses}
        self.glossary: dict[int, str] = {}
        self.by_surface: dict[str, set[int]] = defaultdict(set)
        self.by_gloss: dict[str, set[int]] = defaultdict(set)
        for text, token_ids in self.strict.by_text.items():
            key = meaning_key(text)
            for token_id in token_ids:
                if key:
                    self.by_surface[key].add(token_id)
                gloss = glossary.get(token_id)
                if gloss is None:
                    continue
                gloss_key = meaning_key(gloss.strip(_GLOSS_PUNCTUATION))
                if gloss_key and gloss_key not in ignored:
                    self.glossary[token_id] = gloss_key
                    self.by_gloss[gloss_key].add(token_id)
        self.aliases = {word: tuple(forms) for word, forms in (aliases or {}).items()}
        self._accepted: dict[tuple[str, bool], frozenset[int]] = {}

    def surface_forms(self, word: str, expand: bool = False) -> frozenset[str]:
        """Folded spellings: the word, its order-ops synonyms and its aliases."""
        forms = [*(synonyms(word) if expand else [word]), *self.aliases.get(word, ())]
        return frozenset(key for form in forms if (key := meaning_key(form)))

    def meanings(self, word: str, expand: bool = False) -> frozenset[str]:
        """Gloss keys: the word, its number word, aliases, glosses of its tokens."""
        forms = [word, *self.aliases.get(word, ())]
        if expand and word.isdigit():
            forms += synonyms(word)
        keys = {meaning_key(form) for form in forms}
        keys.update(
            self.glossary[token_id]
            for token_id in self.strict(word)
            if token_id in self.glossary
        )
        keys.discard("")
        return frozenset(keys)

    def __call__(self, word: str, expand: bool = False) -> frozenset[int]:
        if (word, expand) not in self._accepted:
            ids = set(self.strict(word, expand))
            for key in self.surface_forms(word, expand):
                ids.update(self.by_surface.get(key, ()))
            for key in self.meanings(word, expand):
                # Tokens glossed as the meaning, or spelling it (黄 -> " yellow").
                ids.update(self.by_gloss.get(key, ()))
                ids.update(self.by_surface.get(key, ()))
            self._accepted[word, expand] = frozenset(ids)
        return self._accepted[word, expand]

    def translated(self, word: str, expand: bool = False) -> frozenset[int]:
        """Accepted IDs that are not strict spellings of the word."""
        return self(word, expand) - self.strict(word, expand)

    def source(self, word: str, token_id: int, expand: bool = False) -> str | None:
        """``"strict"``, ``"translated"`` or ``None`` (not accepted)."""
        if token_id in self.strict(word, expand):
            return "strict"
        return "translated" if token_id in self(word, expand) else None
