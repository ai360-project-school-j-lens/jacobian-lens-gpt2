"""Architecture-independent lens readouts and dataset evaluation.

Models implement :class:`jlens.protocol.LensModel`; use ``jlens.from_hf``
for HuggingFace causal language models. Each lens reads pre-final-norm block
outputs and calls ``model.unembed`` (final norm, head, and model-specific
logit transforms). Token spelling lookup excludes tokenizer-added specials;
behaviour metrics exclude special-token positions, without assuming a BOS.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from tqdm.auto import tqdm

from jlens.fitting import fit
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.readout import (
    LensReadout,
    as_readout,
    readout,
    selected_token_ranks,
    top_token_ids,
    warn_legacy_readout,
)
from jlens.vis import SliceData, build_page, notebook_iframe

Lens = Callable[[str], torch.Tensor | LensReadout]

DATASETS = ["multihop", "multilingual", "order-ops", "poetry", "association", "typo"]

# порядок intermediates внутри пункта (см. data/evaluations)
ROLE_NAMES = {
    "multilingual": ["language", "relation", "source word", "answer (English)"],
    "order-ops": ["intermediate number", "operation"],
}


def load_eval(repo_dir: str, dataset: str) -> list[dict]:
    """Items of ``data/evaluations/lens-eval-{dataset}.json``."""
    path = os.path.join(repo_dir, "data", "evaluations", f"lens-eval-{dataset}.json")
    with open(path) as f:
        return json.load(f)["items"]


# --------------------------------------------------------------------------- #
# Fit corpus
# --------------------------------------------------------------------------- #

def _detokenize_wikitext(text: str) -> str:
    """WikiText-103 raw keeps the tokenizer artifacts: "2 @.@ 4 km", "well @-@ known", "1 @,@ 000"."""
    for artifact, char in [(" @.@ ", "."), (" @-@ ", "-"), (" @,@ ", ",")]:
        text = text.replace(artifact, char)
    return text


# J_l — среднее якобиана по корпусу, поэтому корпус должен покрывать разные жанры, а не только
# энциклопедический WikiText. source: (path, config, split, text(record), stride — берём каждую stride-ю запись,
# чтобы не набрать подряд абзацы одной статьи)
FIT_SOURCES = {
    "web": ("HuggingFaceFW/fineweb", "sample-10BT", "train", lambda r: r["text"], 1),
    "wikipedia": ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", lambda r: _detokenize_wikitext(r["text"]), 20),
    "fiction": ("roneneldan/TinyStories", None, "train", lambda r: r["text"], 5),
    "dialogue": (
        "HuggingFaceH4/ultrachat_200k", None, "train_sft",
        lambda r: "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in r["messages"]), 5,
    ),
    "math": ("openai/gsm8k", "main", "train", lambda r: f"Question: {r['question']}\nAnswer: {r['answer']}", 5),
    "code": ("openai/openai_humaneval", None, "test", lambda r: r["prompt"] + r["canonical_solution"], 4),
    "poetry": ("merve/poetry", None, "train", lambda r: r["content"], 5),
    "spanish": ("wikimedia/wikipedia", "20231101.es", "train", lambda r: r["text"], 5),
    "french": ("wikimedia/wikipedia", "20231101.fr", "train", lambda r: r["text"], 5),
    "german": ("wikimedia/wikipedia", "20231101.de", "train", lambda r: r["text"], 5),
    "portuguese": ("wikimedia/wikipedia", "20231101.pt", "train", lambda r: r["text"], 5),
}

# число промптов из каждого источника
FIT_MIX = {
    "web": 150,
    "wikipedia": 60,
    "fiction": 60,
    "dialogue": 60,
    "math": 60,
    "code": 40,
    "poetry": 40,
    "spanish": 15,
    "french": 15,
    "german": 15,
    "portuguese": 15,
}


def load_fit_prompts(mix: dict[str, int] = FIT_MIX, *, min_chars: int = 200, seed: int = 0) -> pd.DataFrame:
    """Prompts for ``jlens.fit`` from the corpora of :data:`FIT_SOURCES` (streamed from the HF Hub).

    A row per prompt (columns ``source``, ``text``), shuffled with `seed` so that a partial
    fit checkpoint already covers every source. A source that fails to load is skipped with
    a message, so check ``source.value_counts()``.
    """
    from datasets import load_dataset

    rows = []
    bar = tqdm(total=sum(mix.values()), desc="fit corpus", unit="prompt")
    for source, n in mix.items():
        path, config, split, text_of, stride = FIT_SOURCES[source]
        bar.set_postfix(source=source)
        texts = []
        try:
            seen = 0
            for record in load_dataset(path, config, split=split, streaming=True):
                text = text_of(record).replace("\r\n", "\n").strip()
                if len(text) < min_chars:
                    continue
                if seen % stride == 0:
                    texts.append(text)
                    bar.update()
                    if len(texts) == n:
                        break
                seen += 1
        except Exception as error:
            print(f"{source}: skipped after {len(texts)} prompts ({type(error).__name__}: {error})")
        rows += [{"source": source, "text": text} for text in texts]
    bar.close()
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


class _FitProgress(logging.Handler):
    """Moves a tqdm bar along the per-prompt log records of ``jlens.fit``."""

    def __init__(self, bar: tqdm) -> None:
        super().__init__(logging.INFO)
        self.bar = bar

    def emit(self, record: logging.LogRecord) -> None:
        message = str(record.msg)
        if message.startswith("  jacobian forward:"):
            seq_len, dim_batch, n_passes = record.args
            self.bar.set_postfix(phase="forward", seq_len=seq_len, dim_batch=dim_batch, backwards=n_passes)
        elif message.startswith("  jacobian backward:"):
            done, total = record.args
            self.bar.set_postfix(phase=f"backward {done}/{total}")
        elif message.startswith("  prompt "):
            done, _, seq_len, _, seconds, _, mean_change = record.args
            self.bar.update(done - self.bar.n)
            self.bar.set_postfix(seq_len=seq_len, sec=f"{seconds:.0f}", d_mean=f"{mean_change:.1e}")
        elif message.startswith("  resuming"):
            self.bar.update(record.args[0] - self.bar.n)
        elif message.startswith("  skipping prompt"):
            self.bar.update(record.args[0] + 1 - self.bar.n)


def fit_with_progress(model: LensModel, prompts: Sequence[str], **fit_kwargs) -> JacobianLens:
    """``jlens.fit`` with a tqdm bar over prompts.

    Postfix: seconds per prompt and ``d_mean`` — relative shift of the running mean of J
    (falls ~1/n once the lens has converged).
    """
    # fit пишет строку лога на каждый промпт — по ним и двигаем бар
    logger = logging.getLogger("jlens.fitting")
    level = logger.level
    logger.setLevel(logging.INFO)
    with tqdm(total=len(prompts), desc="fit J-lens", unit="prompt") as bar:
        handler = _FitProgress(bar)
        logger.addHandler(handler)
        try:
            return fit(model, prompts, **fit_kwargs)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)


# --------------------------------------------------------------------------- #
# Lenses
# --------------------------------------------------------------------------- #


ActivationReadout = Callable[
    [LensModel, dict[int, torch.Tensor]], torch.Tensor | LensReadout
]


def _stack_readouts(values: list[LensReadout]) -> LensReadout:
    # Copy/cast directly into the output, without a full fp32 list plus stack.
    logits = torch.empty(
        (len(values), *values[0].logits.shape),
        device=values[0].logits.device, dtype=torch.float32,
    )
    for layer, value in enumerate(values):
        logits[layer].copy_(value.logits)
    scores = (
        logits if all(
            v.logits is v.ranking_scores and v.logits.dtype != torch.float64
            for v in values
        ) else torch.stack([v.ranking_scores for v in values])
    )
    return LensReadout(logits, scores)


@torch.no_grad()
def logit_lens_from_activations(
    model: LensModel, activations: dict[int, torch.Tensor], *,
    return_readout: bool = False,
) -> torch.Tensor | LensReadout:
    """Baseline readout; opt into both lexical scores and distribution logits.

    The default tensor remains the actual model-distribution logits for public
    compatibility. Pass ``return_readout=True`` for lexical evaluation/display.
    """
    if not return_readout:
        return torch.stack([
            model.unembed(activations[layer][0].float()).float()
            for layer in range(model.n_layers)
        ])
    return _stack_readouts([
        readout(model, activations[layer][0].float())
        for layer in range(model.n_layers)
    ])


@torch.no_grad()
def logit_lens(
    model: LensModel, text: str, *, return_readout: bool = False,
) -> torch.Tensor | LensReadout:
    """Read block outputs; use ``return_readout=True`` for lexical evaluation."""
    with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
        model.forward(model.encode(text))
    return logit_lens_from_activations(
        model, recorder.activations, return_readout=return_readout,
    )


@torch.no_grad()
def jacobian_lens(
    model: LensModel, lens: JacobianLens, text: str, *, return_readout: bool = False,
) -> torch.Tensor | LensReadout:
    """Transport then decode; opt into dual spaces with ``return_readout=True``."""
    # активации снимаются теми же хуками на model.layers, что и при fit,
    # поэтому J_l применяется ровно к тому, на чём он обучен
    with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
        model.forward(model.encode(text))
    logits = []
    for layer in range(model.n_layers):
        residual = recorder.activations[layer][0].float()
        # у последнего блока J = I — это логиты модели
        if layer != model.n_layers - 1 and layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits.append(
            readout(model, residual) if return_readout else model.unembed(residual).float()
        )
    return _stack_readouts(logits) if return_readout else torch.stack(logits)


def identity_lens(model: LensModel) -> JacobianLens:
    """JacobianLens with J_l = I: through :func:`jacobian_lens` it must equal the logit lens."""
    eye = torch.eye(model.d_model)
    return JacobianLens(
        {layer: eye for layer in range(model.n_layers - 1)}, n_prompts=0, d_model=model.d_model
    )


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


def shown_layers(n_layers: int, layer_stride: int = 1) -> list[int]:
    """Every `layer_stride`-th layer plus the last one (the model output)."""
    return sorted(set(range(0, n_layers, layer_stride)) | {n_layers - 1})


def lens_slice(model: LensModel, lens: Lens, prompt: str, top_n: int = 10, layer_stride: int = 1) -> SliceData:
    """SliceData for build_page; every token that appears in some top-N cell is tracked."""
    token_ids = model.encode(prompt)[0].tolist()
    result = lens(prompt)
    if not isinstance(result, LensReadout):
        warn_legacy_readout(model)
    logits = as_readout(result).ranking_scores
    del result  # Displays do not need to retain distribution logits.
    layers = shown_layers(logits.shape[0], layer_stride)
    seq_len, n_layers, vocab_size = logits.shape[1], len(layers), logits.shape[-1]

    if not 0 <= top_n <= vocab_size:
        raise ValueError("top_n must be between zero and vocabulary size")
    # Work on layer views; never copy or sort the full layer/position vocabulary.
    top_ids = torch.stack([top_token_ids(logits[layer], top_n).cpu() for layer in layers], dim=1)
    tracked = sorted(set(top_ids.flatten().tolist()))
    tracked_ids = torch.tensor(tracked, dtype=torch.long, device=logits.device)
    rank_tensor = torch.stack([
        (selected_token_ranks(logits[layer], tracked_ids) - 1).cpu() for layer in layers
    ], dim=1)

    def decode(token):
        return model.tokenizer.decode([token], clean_up_tokenization_spaces=False)
    return SliceData(
        seq_len=seq_len,
        layers=layers,
        context_token_ids=token_ids,
        context_token_strs=[decode(t) for t in token_ids],
        top_ids=top_ids.numpy().astype("int32"),
        top_ranks=torch.arange(top_n).expand(seq_len, n_layers, top_n).numpy().astype("int32"),
        tracked_token_ids=tracked,
        rank_tensor=rank_tensor.numpy().astype("int32"),
        vocab_fragment={t: decode(t) for t in set(tracked) | set(token_ids)},
        vocab_size=vocab_size,
    )


def show_lens(
    model: LensModel, lens: Lens, prompt: str, title: str = "Logit lens", description: str = "", layer_stride: int = 1
) -> None:
    """Interactive position x layer view of the lens top-N (``build_page`` from jlens.vis)."""
    slice_data = lens_slice(model, lens, prompt, layer_stride=layer_stride)
    page, _, _ = build_page(slice_data, prompt, title=title, description=description)
    display(notebook_iframe(page))


def top_tokens_table(
    model: LensModel,
    lenses: dict[str, Lens],
    prompt: str,
    position: int = -1,
    top_n: int = 5,
    layer_stride: int = 1,
) -> pd.DataFrame:
    """Top-N tokens of every lens at one position, a row per shown layer (last row = model output)."""
    layers = shown_layers(model.n_layers, layer_stride)
    columns = {}
    for name, lens in lenses.items():
        result = lens(prompt)
        if not isinstance(result, LensReadout):
            warn_legacy_readout(model)
        scores = as_readout(result).ranking_scores
        del result
        top = [top_token_ids(scores[layer, position], top_n).tolist() for layer in layers]
        columns[name] = [[model.tokenizer.decode([t]) for t in row] for row in top]
        del scores
    return pd.DataFrame(columns, index=pd.Index(layers, name="layer"))


# --------------------------------------------------------------------------- #
# Spellings, readout position, ranks
# --------------------------------------------------------------------------- #

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

OPERATION_SYNONYMS = {
    "addition": ["+", "plus", "add", "sum"],
    "subtraction": ["-", "minus", "subtract", "difference"],
    "multiplication": ["*", "×", "times", "multiply", "product"],
    "division": ["/", "÷", "divided", "divide"],
    "squared": ["²", "^", "square"],
    "mod": ["%", "modulo", "remainder"],
}


def synonyms(word: str) -> list[str]:
    """Synonym set of an order-ops intermediate: numbers -> digits and words, operations -> symbols and words."""
    if word.isdigit() and int(word) < 100:
        tens, ones = divmod(int(word), 10)
        spelled = _ONES[int(word)] if int(word) < 20 else _TENS[tens] + ("" if ones == 0 else "-" + _ONES[ones])
        return [word, spelled]
    return [word, *OPERATION_SYNONYMS.get(word, [])]


def single_token_ids(tokenizer, word: str, expand: bool = False) -> set[int]:
    """Complete single-token spellings, with optional space and case variants.

    Encoding must round-trip exactly; unknown/special tokens and whitespace
    alone are not lexical spellings. Unlike ``DecodedSpellings``, this only
    finds IDs returned by encoding, not every equivalent decoded vocabulary ID.
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    special_ids.update(
        token for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (token := getattr(tokenizer, attr, None)) is not None
    )
    ids = set()
    for spelling in synonyms(word) if expand else [word]:
        if not spelling or spelling.isspace():
            continue
        for variant in {spelling, spelling.lower(), spelling.capitalize()}:
            for text in (variant, " " + variant):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if (
                    len(tokens) == 1
                    and tokens[0] not in special_ids
                    and tokenizer.decode(
                        tokens, clean_up_tokenization_spaces=False,
                    ) == text
                ):
                    ids.add(tokens[0])
    return ids


_FUZZY_FORMS: dict[int, dict[str, set[int]]] = {}


def _forms_by_text(tokenizer) -> dict[str, set[int]]:
    """Vocabulary index: stripped, case-folded decoded text -> every id with that text.

    Built by scanning the whole vocabulary once per tokenizer, so it finds spellings
    :func:`single_token_ids` cannot construct (odd casings, byte-level variants, scripts
    where the leading-space form is not a simple concatenation). Cached by tokenizer id.
    """
    key = id(tokenizer)
    if key not in _FUZZY_FORMS:
        index: dict[str, set[int]] = {}
        for token, token_id in tokenizer.get_vocab().items():
            text = tokenizer.convert_tokens_to_string([token]).strip().casefold()
            if text:
                index.setdefault(text, set()).add(token_id)
        _FUZZY_FORMS[key] = index
    return _FUZZY_FORMS[key]


def fuzzy_spelling_ids(
    tokenizer, word: str, expand: bool = False, *, prefix: bool = True
) -> set[int]:
    """Token ids that count as the model saying ``word``, matched leniently.

    Three arms, in order:

    1. **Exact.** Every vocabulary token whose decoded text equals ``word`` ignoring
       surrounding whitespace and case. This is what makes ``' Italy'`` and ``'Italy'``
       one answer rather than two, and it is a superset of :func:`single_token_ids`.
    2. **Prefix** (only when the exact arm is empty, i.e. ``word`` has no single-token
       form, and only when ``prefix`` is True). The first token of the word's own
       tokenization, accepted when it decodes to a non-whitespace prefix of ``word`` of
       at least two characters -- or one character if that character is non-ASCII, since
       a CJK token carries far more of the word than a Latin letter does. This is what
       makes ``tellurium``, ``automne`` and ``火曜日`` scorable at all.
    3. **Unscorable.** Otherwise the empty set, which :func:`_readout_rows` records as
       ``ranks=None`` so the word is *excluded* rather than scored. Qwen tokenizes a
       numeral like ``26`` as ``2``+``6`` and ``" 26"`` as ``" "``+``"26"``, so
       :func:`spelling_ids` would fall back to a bare space -- an extremely common next
       token, which scores as a hit on any prompt that happens to continue with
       whitespace.

    The prefix arm raises absolute hit rates for every lens, because a word's first
    token is more probable than the whole word. It is applied identically to both
    lenses, so lens-vs-lens contrasts stay fair, but absolute numbers are not comparable
    to a strict run. Pass ``prefix=False`` for the exact arm alone.
    """
    spellings = synonyms(word) if expand else [str(word)]
    index = _forms_by_text(tokenizer)
    ids: set[int] = set()
    for spelling in spellings:
        ids |= index.get(str(spelling).strip().casefold(), set())
    if ids or not prefix:
        return ids
    for spelling in spellings:
        text = str(spelling).strip()
        if not text:
            continue
        for candidate in (" " + text, text):
            tokens = tokenizer.encode(candidate, add_special_tokens=False)
            if not tokens:
                continue
            piece = tokenizer.decode([tokens[0]]).strip()
            enough = len(piece) >= 2 or (len(piece) == 1 and not piece.isascii())
            if piece and enough and text.casefold().startswith(piece.casefold()):
                return {tokens[0]}
    return set()


def spelling_ids(tokenizer, word: str, expand: bool = False) -> set[int]:
    """:func:`single_token_ids`, or the first token of " word" if there are none."""
    # слово не помещается в один токен — берём первый токен варианта с пробелом
    return single_token_ids(tokenizer, word, expand) or {tokenizer.encode(" " + word, add_special_tokens=False)[0]}


def readout_position(tokenizer, token_ids: Sequence[int], dataset: str) -> int:
    """Readout position from data/evaluations/README.md.

    poetry — последний перевод строки (конец первой строки куплета);
    остальные — последний токен промпта (токен перед target / закрывающая точка / хвост опечатки).
    """
    if dataset == "poetry":
        newlines = [i for i, t in enumerate(token_ids) if "\n" in tokenizer.decode([t])]
        if newlines:
            return newlines[-1]
    return len(token_ids) - 1


def token_ranks(logits: torch.Tensor, token_ids: set[int]) -> torch.Tensor:
    """1-based lexical ranks, descending score then ascending token ID.

    Supply pre-softcap scores, not distribution logits, when available. Legacy
    tensors are accepted but cannot recover ordering lost to saturation.
    """
    ids = torch.tensor(sorted(token_ids), device=logits.device, dtype=torch.long)
    return selected_token_ranks(logits, ids)


# --------------------------------------------------------------------------- #
# Readout evaluation (data/evaluations protocol)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_readout(model: LensModel, lens: Lens, dataset: str, items: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `lens` over the dataset prompts and read it out at the readout position.

    Returns ``(words, items)``:

    Supply a lens returning ``LensReadout`` (e.g. built-in helpers with
    ``return_readout=True``) for pre-softcap lexical ordering. Legacy tensors
    remain distribution logits and cannot recover ordering lost to saturation.
    Exact rank ties are resolved by ascending token ID, not optimistic ties.
    Behavior/argmax columns always describe the actual distribution.

    * ``words`` — a row per (item, word). ``kind`` is ``intermediate``, ``target`` or
      ``control`` (intermediates of another item of the same dataset — a chance baseline).
      ``ranks`` is the 1-based rank at every layer, min over spellings (for order-ops —
      over the synonym set).
    Targets use complete single-token spellings only, including order-ops
    synonyms, for both ranks and correctness. Unsupported multi-token targets
    have null ranks/correctness, not prefix hits. Intermediate/control probes
    retain the historical first-token fallback in :func:`spelling_ids`.

    * ``items`` — a row per item: the model's answer, the lens top-1 at the readout position
      at every layer, and over all prompt positions: agreement of the layer top-1 with the
      model's top-1, the share of positions where the layer top-1 is the input token itself
      (``copy_rate``) and KL(model || layer).
    """
    word_rows, item_rows = [], []
    for i, item in enumerate(tqdm(items, desc=dataset, leave=False)):
        prompt = item["prompt"].rstrip()
        ids = model.encode(prompt)[0].tolist()
        rows, row = _readout_rows(model, lens(prompt), dataset, items, i, ids)
        word_rows.extend(rows)
        item_rows.append(row)
    return pd.DataFrame(word_rows), pd.DataFrame(item_rows)


def _readout_rows(
    model: LensModel,
    logits: torch.Tensor | LensReadout | Callable[[int], LensReadout],
    dataset: str,
    items: list[dict],
    i: int,
    ids: list[int],
    *,
    spelling_lookup: Callable[[str, bool], set[int]] | None = None,
    preserve_prompt_whitespace: bool = False,
) -> tuple[list[dict], dict]:
    """Score one layer at a time; a callable avoids retaining full dual readouts."""
    if callable(logits):
        layer_readout = logits
        n_layers = model.n_layers
    else:
        if not isinstance(logits, LensReadout):
            warn_legacy_readout(model)
        result = as_readout(logits)
        n_layers = result.logits.shape[0]

        def layer_readout(layer):
            return LensReadout(result.logits[layer], result.ranking_scores[layer])
    tok = model.tokenizer
    expand = dataset == "order-ops"
    item = items[i]
    prompt = item["prompt"] if preserve_prompt_whitespace else item["prompt"].rstrip()
    lookup = spelling_lookup or (lambda word, expand: spelling_ids(tok, word, expand))

    def decode(token_ids):
        if spelling_lookup is not None:
            return tok.decode(token_ids, clean_up_tokenization_spaces=False)
        return tok.decode(token_ids)

    word_rows = []
    position = readout_position(tok, ids, dataset)
    word_token_ids = []

    other = items[(i + len(items) // 2) % len(items)]["intermediates"]
    words = [("intermediate", word, role) for role, word in enumerate(item["intermediates"])]
    words += [("control", word, role) for role, word in enumerate(other) if word not in item["intermediates"]]
    if "target" in item:
        words.append(("target", item["target"], 0))
    special_ids = set(getattr(tok, "all_special_ids", []) or [])
    special_ids.update(
        token for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
        if (token := getattr(tok, attr, None)) is not None
    )
    positions = [position for position, token in enumerate(ids) if token not in special_ids]
    if not positions:
        raise ValueError(f"{dataset}/{item['name']}: no non-special prompt tokens")
    prompt_ids = {ids[position] for position in positions}
    target_ids = set()
    for kind, word, role in words:
        if kind == "target" and spelling_lookup is None:
            # Prefix probes are useful for intermediates, not completed answers.
            word_ids = single_token_ids(tok, word, expand)
        else:
            word_ids = lookup(word, expand)
        if kind == "target":
            target_ids = word_ids
        word_token_ids.append(word_ids)
        word_rows.append({
            "dataset": dataset,
            "item": item["name"],
            "kind": kind,
            "word": word,
            "role": role,
            "single_token": (
                bool(word_ids) if spelling_lookup is not None
                else bool(single_token_ids(tok, word, expand))
            ),
            "in_prompt": bool(word_ids & prompt_ids),
        })

    # Retain final probabilities plus just one layer's distribution workspace.
    final = layer_readout(n_layers - 1)
    final_top = final.logits[positions].argmax(-1)
    final_logp = final.logits[positions].float().log_softmax(-1)
    final_prob = final_logp.exp()
    model_top1 = int(final.logits[position].argmax())
    input_ids = torch.tensor([ids[p] for p in positions], device=final_top.device)
    agreements, copy_rates, kls, readout_tops, layer_ranks = [], [], [], [], []
    for layer in range(n_layers):
        current = final if layer == n_layers - 1 else layer_readout(layer)
        layer_ranks.append([
            token_ranks(current.ranking_scores[position], word_ids).min()
            if word_ids else torch.tensor(-1, device=final_top.device)
            for word_ids in word_token_ids
        ])
        top = current.logits[positions].argmax(-1)
        logp = final_logp if layer == n_layers - 1 else current.logits[positions].float().log_softmax(-1)
        kls.append((final_prob * (final_logp - logp)).sum(-1).mean())
        agreements.append((top == final_top).float().mean())
        copy_rates.append((top == input_ids).float().mean())
        readout_tops.append(current.logits[position].argmax())
        del current, logp, top
    if word_rows:
        ranks_by_word = torch.stack([torch.stack(r) for r in layer_ranks]).T.cpu().numpy()
        for row, word_ids, ranks in zip(word_rows, word_token_ids, ranks_by_word, strict=True):
            row.update(
                ranks=ranks if word_ids else None,
                best_rank=int(ranks.min()) if word_ids else None,
                best_layer=int(ranks.argmin()) if word_ids else None,
            )
    # Reuse the target rank's accepted IDs, including arithmetic synonyms.
    item_row = {
        "dataset": dataset,
        "item": item["name"],
        "prompt": prompt,
        "target": item.get("target"),
        "readout_token": decode([ids[position]]),
        "model_top1": decode([model_top1]),
        "model_correct": model_top1 in target_ids if target_ids else None,
        "readout_top1": [decode([t]) for t in torch.stack(readout_tops).tolist()],
        "agreement": torch.stack(agreements).cpu().numpy(),
        "copy_rate": torch.stack(copy_rates).cpu().numpy(),
        "kl_to_final": torch.stack(kls).cpu().numpy(),
    }
    return word_rows, item_row


def evaluate_all(
    model: LensModel, lens: Lens, evals: dict[str, list[dict]], desc: str = "datasets"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """:func:`evaluate_readout` over every dataset of `evals`, concatenated."""
    frames = [evaluate_readout(model, lens, dataset, items) for dataset, items in tqdm(evals.items(), desc=desc)]
    return (
        pd.concat([words for words, _ in frames], ignore_index=True),
        pd.concat([items for _, items in frames], ignore_index=True),
    )


@torch.no_grad()
def evaluate_paired(
    model: LensModel,
    lens: JacobianLens,
    evals: dict[str, list[dict]],
    desc: str = "paired lenses",
    *,
    logit_readout: ActivationReadout | None = None,
    layer_logit_readout: Callable[..., LensReadout] | None = None,
    spelling_lookup: Callable[[str, bool], set[int]] | None = None,
    preserve_prompt_whitespace: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate logit lens and J-lens from one set of block outputs per prompt.

    Returns the same metrics as :func:`evaluate_all`, with a ``lens`` column
    (``logit lens`` or ``J-lens``). Both use the exact same pre-final-norm residuals
    and final model logits. Only the J-lens transports inner-layer residuals.
    ``logit_readout(model, activations)`` can supply an explicit handwritten
    baseline, returning a ``LensReadout`` of ``[n_layers, seq_len, vocab]``.
    Its final row is set to the shared model output. Legacy tensor callbacks
    remain supported: BOTH lenses then rank distribution logits, with a warning
    for known softcapped models. Use dual readouts to avoid saturation ordering
    loss. Ties in either space are broken by ascending token ID.
    All inner layers must be fitted; silently substituting the logit lens for
    a missing Jacobian would make the comparison misleading.

    Activations are released after each prompt. Built-in readouts stream one
    layer at a time, retaining the final model readout for comparison. Custom
    callbacks may return full tensors; these are neither cloned nor mutated.
    Alternatively, ``layer_logit_readout(model, activations, layers=[layer])``
    returns a dual ``LensReadout`` of shape ``[1, seq_len, vocab]`` for each
    inner baseline layer, without stacking vocabulary tensors across layers.
    The two callback options are mutually exclusive; neither overrides the
    shared final model readout. Dataset-wide vocabulary logits are not cached.

    Optional ``spelling_lookup(word, expand)`` overrides lexical scoring for
    both target ranks and correctness. Empty sets yield null ranks/correctness;
    callers must report coverage. Summary helpers exclude null ranks from their
    denominators; an entirely unsupported pass@k group has a NaN score.
    ``preserve_prompt_whitespace=True`` disables legacy prompt stripping.
    Default intermediate/control probes retain the historical prefix fallback;
    default targets require complete single-token spellings. Multi-token-only
    targets are unscorable (null), not incorrect or prefix-correct. This is
    next-token lexical scoring, not generated multi-token answer accuracy.
    """
    if logit_readout is not None and layer_logit_readout is not None:
        raise ValueError("use only one of logit_readout and layer_logit_readout")
    if lens.d_model != model.d_model:
        raise ValueError("lens d_model does not match the model")
    missing = set(range(model.n_layers - 1)) - set(lens.source_layers)
    if missing:
        raise ValueError(f"J-lens is missing inner layers {sorted(missing)}")

    word_rows, item_rows = [], []
    for dataset, samples in tqdm(evals.items(), desc=desc):
        for i, item in enumerate(tqdm(samples, desc=dataset, leave=False)):
            prompt = item["prompt"] if preserve_prompt_whitespace else item["prompt"].rstrip()
            input_ids = model.encode(prompt)
            ids = input_ids[0].tolist()
            with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
                model.forward(input_ids)
            # The final raw block output goes through the adapter's complete
            # readout, including any model-specific logit softcap.
            final_readout = readout(model, recorder.activations[model.n_layers - 1][0].float())
            legacy_scores = False
            for name in ("logit lens", "J-lens"):
                callback_result = None
                if name == "logit lens" and logit_readout is not None:
                    value = logit_readout(model, recorder.activations)
                    legacy_scores = not isinstance(value, LensReadout)
                    if legacy_scores:
                        warn_legacy_readout(model)
                    callback_result = as_readout(value)
                    del value
                    expected = (model.n_layers, *final_readout.logits.shape)
                    if (callback_result.logits.shape != expected
                            or callback_result.ranking_scores.shape != expected):
                        raise ValueError("logit_readout must return [n_layers, seq_len, vocab]")

                def layer_readout(
                    layer, final_readout=final_readout, callback_result=callback_result,
                    activations=recorder.activations, name=name, legacy_scores=legacy_scores,
                ):
                    if layer == model.n_layers - 1:
                        value = final_readout
                    elif name == "logit lens" and layer_logit_readout is not None:
                        value = layer_logit_readout(model, activations, layers=[layer])
                        if not isinstance(value, LensReadout):
                            raise TypeError("layer_logit_readout must return LensReadout")
                        expected = (1, *final_readout.logits.shape)
                        if (value.logits.shape != expected
                                or value.ranking_scores.shape != expected):
                            raise ValueError("layer_logit_readout must return [1, seq_len, vocab]")
                        value = LensReadout(value.logits[0], value.ranking_scores[0])
                    elif callback_result is not None:
                        value = LensReadout(
                            callback_result.logits[layer], callback_result.ranking_scores[layer],
                        )
                    else:
                        residual = activations[layer][0].float()
                        if name == "J-lens":
                            residual = lens.transport(residual, layer)
                        value = readout(model, residual)
                    # Legacy callbacks compare both lenses in their available space.
                    return LensReadout(value.logits, value.logits) if legacy_scores else value

                rows, row = _readout_rows(
                    model, layer_readout, dataset, samples, i, ids,
                    spelling_lookup=spelling_lookup,
                    preserve_prompt_whitespace=preserve_prompt_whitespace,
                )
                word_rows.extend({**entry, "lens": name} for entry in rows)
                item_rows.append({**row, "lens": name})
                del callback_result, layer_readout
            del recorder, final_readout
    return pd.DataFrame(word_rows), pd.DataFrame(item_rows)


def _layers(n_layers: int, layers: slice | Sequence[int] | None) -> np.ndarray:
    return np.arange(n_layers)[slice(None) if layers is None else layers]


def pass_at_k(words: pd.DataFrame, ks: Sequence[int] = (1, 5, 10, 100), layers: slice | Sequence[int] | None = None) -> pd.DataFrame:
    """pass@k as in data/evaluations: mean over items of the fraction of words whose min-over-layers rank <= k.

    ``layers`` restricts the min, e.g. ``slice(None, -1)`` excludes model output.
    Null ranks are excluded, not misses; items without ranked words are omitted.
    Entirely unsupported groups retain a NaN score. Coverage columns count words
    and items before/after exclusion. Legacy intermediate prefix probes remain
    ranked even when ``single_token`` is false.
    """
    keys = [c for c in ("lens", "dataset", "kind") if c in words]
    counts = ["n_words", "n_words_scored", "n_items", "n_items_scored"]
    rows = []
    for key, group in words.groupby(keys):
        scored = group[group["ranks"].notna()]
        coverage = dict(zip(counts, [
            len(group), len(scored), group["item"].nunique(),
            scored["item"].nunique(),
        ], strict=True))
        if not scored.empty:
            ranks = np.stack(scored["ranks"])
            best = ranks[:, _layers(ranks.shape[1], layers)].min(1)
        for k in ks:
            score = (
                pd.Series(best <= k, index=scored.index)
                .groupby(scored["item"]).mean().mean()
                if not scored.empty else np.nan
            )
            rows.append({**dict(zip(keys, key, strict=False)), "k": k,
                         "score": score, **coverage})
    return pd.DataFrame(rows, columns=[*keys, "k", "score", *counts])


def layer_hit_rate(words: pd.DataFrame, k: int) -> pd.DataFrame:
    """Share of ranked words with rank <= k: [dataset x layer].

    Null ranks are excluded; datasets with no ranked words are omitted. Report
    coverage separately (e.g. using :func:`pass_at_k`), not as zero hit rates.
    """
    scored = words[words["ranks"].notna()]
    return pd.DataFrame({d: (np.stack(g["ranks"]) <= k).mean(0) for d, g in scored.groupby("dataset", sort=False)}).T


def layer_median_rank(words: pd.DataFrame) -> pd.DataFrame:
    """Median of non-null ranks: [dataset x layer].

    Datasets with no ranked words are omitted, as in :func:`layer_hit_rate`.
    """
    scored = words[words["ranks"].notna()]
    return pd.DataFrame({d: np.median(np.stack(g["ranks"]), 0) for d, g in scored.groupby("dataset", sort=False)}).T


def layer_mean(items: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean of a per-layer column of the items frame: [dataset x layer]."""
    return pd.DataFrame({d: np.stack(g[column]).mean(0) for d, g in items.groupby("dataset", sort=False)}).T


def first_layer(curve: pd.Series, threshold: float, above: bool = True) -> int | None:
    """First layer where `curve` crosses `threshold` (>= if `above`, else <=), None if never."""
    hit = curve >= threshold if above else curve <= threshold
    return int(curve.index[hit.values.argmax()]) if hit.any() else None


def plot_layer_curves(
    curves: dict[str, pd.DataFrame],
    title: str,
    ylabel: str,
    logy: bool = False,
    ncols: int = 3,
) -> None:
    """A subplot per dataset, a line per entry of `curves` (each [dataset x layer])."""
    datasets = list(next(iter(curves.values())).index)
    # Assign colors before skipping absent series so every panel matches the legend.
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {label: palette[i % len(palette)] for i, label in enumerate(curves)}
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), sharex=True, squeeze=False)
    for ax, dataset in zip(axes.flat, datasets, strict=False):
        for label, frame in curves.items():
            if dataset in frame.index:
                ax.plot(frame.columns, frame.loc[dataset], marker="o", ms=3, label=label,
                        color=colors[label], linestyle="--" if "control" in label else "-")
        ax.set_title(dataset)
        ax.set_xlabel("layer (last = model output)")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(datasets):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
