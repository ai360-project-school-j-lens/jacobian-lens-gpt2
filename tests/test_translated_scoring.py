from jlens.translated_scoring import TranslatedSpellings, meaning_key


class _Tokenizer:
    """Whole-token vocabulary with one special token."""

    all_special_ids = [0]
    bos_token_id = eos_token_id = pad_token_id = None

    def __init__(self, tokens):
        self.tokens = tokens

    def get_vocab(self):
        return {text: i for i, text in enumerate(self.tokens)}

    def decode(self, ids, **kwargs):
        return "".join(self.tokens[i] for i in ids)


TOKENS = [
    "<eos>", "Italy", " Italy", "意大利", "二十", "２０", "20", " twenty",
    "黄", " yellow", "黄色", "frag", "除以", " ITALY", "Ital",
]
GLOSS = {3: "Italy", 4: "twenty", 8: "yellow", 10: "yellow", 11: "fragment", 12: "divide by"}


def _spellings(**kwargs):
    tok = _Tokenizer(TOKENS)
    return TranslatedSpellings(tok, vocab_size=len(TOKENS), glossary=GLOSS, **kwargs)


def test_strict_spellings_are_a_subset_and_translations_are_added():
    spellings = _spellings()
    assert spellings.strict("Italy") == {1, 2}
    assert spellings("Italy") == {1, 2, 3, 13}  # gloss and casefolded surface
    assert spellings.translated("Italy") == {3, 13}
    assert spellings.source("Italy", 3) == "translated"
    assert spellings.source("Italy", 1) == "strict"
    assert spellings.source("Italy", 14) is None  # never a prefix


def test_numbers_match_digits_words_full_width_and_chinese():
    spellings = _spellings()
    assert spellings("20") == {5, 6}  # NFKC folds the full-width digits
    # order-ops synonyms: 20 <-> twenty, in English and Chinese.
    assert spellings("20", expand=True) == {4, 5, 6, 7}


def test_foreign_label_maps_through_its_own_gloss():
    spellings = _spellings()
    assert spellings.meanings("黄") == {"黄", "yellow"}
    assert spellings("黄") == {8, 9, 10}


def test_operation_synonyms_are_not_translated():
    tokens = ["<eos>", " product", "产品", "乘法", " multiplication"]
    gloss = {2: "product", 3: "multiplication"}
    spellings = TranslatedSpellings(
        _Tokenizer(tokens), vocab_size=len(tokens), glossary=gloss,
    )
    # " product" is an English synonym; 产品 (a product) is not the operation.
    assert spellings("multiplication", expand=True) == {1, 3, 4}


def test_placeholder_glosses_special_tokens_and_aliases():
    spellings = _spellings(aliases={"division": ["除以"]})
    assert spellings("fragment") == set()
    assert spellings("<eos>") == set()
    assert 12 in spellings("division")
    assert 12 not in _spellings()("division")


def test_meaning_key_folds_width_case_and_space():
    assert meaning_key("  ２０ ") == "20"
    assert meaning_key("Divide  By") == "divide by"
