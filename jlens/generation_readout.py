"""Lens readouts at every position of a model's own greedy continuation.

The paper's intermediate evaluation reads one prompt position. Here the model
first continues the prompt greedily; one forward over prompt + continuation
then reads the J-lens and the logit lens off the same activations at the last
prompt token and at every generated token.

Steps: a continuation of ``m`` tokens ``g_0..g_{m-1}`` has steps ``0..m-1``.
Step ``s`` reads the position at which the model predicts ``g_s``: the last
prompt token for ``s = 0``, the position of ``g_{s-1}`` otherwise. Every
tracked word gets a status per step, from the text alone:

* ``in_text`` — already in the prompt or in ``g_0..g_{s-1}``;
* ``next`` — the continuation from ``g_s`` on starts with the word, i.e. the
  model is about to write it (a next-token readout, trivial for a lens);
* ``latent`` — neither: the condition under which the paper scores an
  intermediate ("neither present in the input nor identical to the predicted
  output").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.readout import readout, selected_token_ranks
from jlens.workspace import _jacobian_on

LENS_ORDER = ("J-lens", "logit lens")
SWEEP_KS = (1, 2, 5, 10, 20, 50, 100)
STATUSES = ("in_text", "next", "latent")
# A word boundary that also works for symbols such as "*" and "+".
_LEFT, _RIGHT = r"(?<![0-9A-Za-z])", r"(?![0-9A-Za-z])"
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*[A-Za-z]|[A-Za-z]")
STOPWORDS = frozenset("""
    a about above after again against all also am an and any are as at be because
    been before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just let like made make many
    may me might more most much must my myself no nor not now of off on once only
    or other our ours ourselves out over own same shall she should so some such
    than that the their theirs them themselves then there these they this those
    through to too under until up upon us very was we were what when where which
    while who whom why will with would yet you your yours yourself yourselves
    one two three still even ever every never always again well into onto
""".split())


@dataclass(frozen=True)
class Continuation:
    """Prompt and greedy continuation token IDs; ``stop`` says why it ended."""

    prompt_ids: list[int]
    generated_ids: list[int]
    stop: str  # "newline", "eos" or "max_new_tokens"

    @property
    def token_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.generated_ids]

    @property
    def start(self) -> int:
        """Sequence index of step 0 (the last prompt token)."""
        return len(self.prompt_ids) - 1

    @property
    def n_steps(self) -> int:
        return len(self.generated_ids)


def cut_continuation(
    tokenizer, generated_ids: Sequence[int], eos_ids: set[int], *, stop_at_newline: bool
) -> tuple[list[int], str]:
    """Keep tokens up to the first EOS or the first newline after text, inclusive.

    The stop token is kept so that the last position of the text is read too
    (the step that predicts the stop), and an immediate EOS still leaves step 0.
    """
    kept: list[int] = []
    for token in generated_ids:
        kept.append(token)
        if token in eos_ids:
            return kept, "eos"
        text = tokenizer.decode(kept, skip_special_tokens=True)
        if stop_at_newline and "\n" in text.lstrip():
            return kept, "newline"
    return kept, "max_new_tokens"


@torch.inference_mode()
def greedy_continuation(
    hf_model,
    tokenizer,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int = 32,
    stop_at_newline: bool = True,
) -> Continuation:
    """Greedy-decode one unpadded prompt and cut it with :func:`cut_continuation`."""
    eos = hf_model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [] if eos is None else [eos])
    device = next(hf_model.parameters()).device
    input_ids = torch.tensor([list(prompt_ids)], device=device)
    output = hf_model.generate(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        pad_token_id=next(iter(eos_ids), tokenizer.pad_token_id),
        eos_token_id=sorted(eos_ids) or None,
    )
    generated = output[0, input_ids.shape[1]:].tolist()
    kept, stop = cut_continuation(
        tokenizer, generated, eos_ids, stop_at_newline=stop_at_newline
    )
    return Continuation(list(prompt_ids), kept, stop)


def word_pattern(spellings: Sequence[str]) -> re.Pattern:
    """Case-insensitive match of any spelling as a whole word (symbols allowed)."""
    words = sorted({s.strip() for s in spellings if s.strip()}, key=len, reverse=True)
    if not words:
        raise ValueError("A tracked word needs at least one non-blank spelling")
    return re.compile(_LEFT + "(?:" + "|".join(map(re.escape, words)) + ")" + _RIGHT, re.I)


def step_texts(
    tokenizer, continuation: Continuation
) -> tuple[list[str], list[str]]:
    """Per step: the text read so far, and the text still to be generated."""
    decode = lambda ids: tokenizer.decode(  # noqa: E731
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    prompt, generated = continuation.prompt_ids, continuation.generated_ids
    contexts = [decode([*prompt, *generated[:s]]) for s in range(len(generated))]
    rests = [decode(generated[s:]) for s in range(len(generated))]
    return contexts, rests


def word_status(context: str, rest: str, pattern: re.Pattern) -> str:
    """``in_text``, ``next`` or ``latent`` for one step; see the module docstring."""
    if pattern.search(context):
        return "in_text"
    match = pattern.search(rest)
    if match is not None and not rest[: match.start()].strip():
        return "next"
    return "latent"


def future_words(
    tokenizer,
    continuation: Continuation,
    *,
    min_letters: int = 4,
    stopwords: frozenset[str] = STOPWORDS,
) -> list[tuple[str, int]]:
    """Content words of the continuation and the step at which each is emitted.

    A word is emitted at the step that generates the token holding its first
    letter. Only first occurrences are kept; words already in the prompt,
    stopwords and words shorter than ``min_letters`` are skipped.
    """
    decode = lambda ids: tokenizer.decode(  # noqa: E731
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    prompt_text = decode(continuation.prompt_ids)
    ends, text = [], ""
    for s in range(continuation.n_steps):
        text = decode(continuation.generated_ids[: s + 1])
        ends.append(len(text))
    words, seen = [], set()
    for match in _WORD.finditer(text):
        word = match.group(0)
        key = word.lower()
        if (
            key in seen or key in stopwords or sum(c.isalpha() for c in word) < min_letters
            or word_pattern([word]).search(prompt_text)
        ):
            continue
        seen.add(key)
        step = next(s for s, end in enumerate(ends) if end > match.start())
        words.append((word, step))
    return words


@torch.inference_mode()
def continuation_ranks(
    model: LensModel,
    lens: JacobianLens,
    continuation: Continuation,
    id_sets: Sequence[Sequence[int]],
) -> np.ndarray:
    """Ranks ``[lens (J, logit), step, layer, word]``, min over each word's IDs.

    One forward over prompt + continuation. Ranks are 1-based over the full
    vocabulary (descending lexical score, ties by token ID). Layers without a
    fitted Jacobian (the final block) are read identically by both lenses.
    Words with no IDs get rank 0, which callers must treat as unsupported.
    """
    flat = sorted({int(t) for ids in id_sets for t in ids})
    column = {token: i for i, token in enumerate(flat)}
    n_layers, n_steps = model.n_layers, continuation.n_steps
    out = np.zeros((2, n_steps, n_layers, len(id_sets)), dtype=np.int32)
    if not flat or not n_steps:
        return out
    input_ids = torch.tensor([continuation.token_ids], device=model.input_device)
    with ActivationRecorder(model.layers, at=range(n_layers)) as recorder:
        model.forward(input_ids)
    rows = slice(continuation.start, continuation.start + n_steps)
    ids = torch.tensor(flat, dtype=torch.long)
    for layer in range(n_layers):
        hidden = recorder.activations[layer][0, rows].float()
        variants = [hidden, hidden]
        if layer in lens.jacobians:
            variants[0] = hidden @ _jacobian_on(lens, layer, hidden.device).T
        for lens_index, residual in enumerate(variants):
            if lens_index == 1 and layer not in lens.jacobians:
                out[1, :, layer] = out[0, :, layer]
                continue
            scores = readout(model, residual).ranking_scores
            ranks = selected_token_ranks(scores, ids.to(scores.device)).cpu().numpy()
            for word, word_ids in enumerate(id_sets):
                if word_ids:
                    out[lens_index, :, layer, word] = ranks[
                        :, [column[int(t)] for t in word_ids]
                    ].min(axis=1)
    return out


def hit_rates(
    frame: pd.DataFrame, by: Sequence[str], *, ks: Sequence[int] = SWEEP_KS,
    rank: str = "best_rank",
) -> pd.DataFrame:
    """Item-weighted hit@k: average hits within each item, then over items.

    ``frame`` has one row per scored (item, word, step); its ``rank`` column is
    the best rank over the layers of interest. Returns one row per ``by`` group
    with columns ``pass@k``, the normalized log-k AUC and item/row counts.
    """
    k_values = np.asarray(ks, dtype=float)
    log_k = np.log(k_values)
    rows = []
    for key, group in frame.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        scores = [
            group.assign(_hit=group[rank] <= k).groupby("item")["_hit"].mean().mean()
            for k in ks
        ]
        y = np.asarray(scores)
        auc = np.sum(np.diff(log_k) * (y[:-1] + y[1:]) / 2) / (log_k[-1] - log_k[0])
        rows.append({
            **dict(zip(by, key, strict=True)),
            **{f"pass@{k}": score for k, score in zip(ks, scores, strict=True)},
            "auc": auc, "n_items": group["item"].nunique(), "n_rows": len(group),
        })
    return pd.DataFrame(rows)


def item_bootstrap_auc_diff(
    frame: pd.DataFrame, *, ks: Sequence[int] = SWEEP_KS, n_boot: int = 1000,
    seed: int = 0, rank: str = "best_rank",
) -> tuple[float, float, float]:
    """Paired item bootstrap of AUC(J-lens) - AUC(logit lens): (diff, lo, hi).

    Items are resampled with replacement and their per-item hit@k curves are
    shared between the lenses, so the interval is paired.
    """
    k_values = np.asarray(ks, dtype=float)
    log_k = np.log(k_values)
    weights = np.diff(log_k) / (log_k[-1] - log_k[0])
    curves = {}
    for lens_name in LENS_ORDER:
        group = frame[frame["lens"].eq(lens_name)]
        curves[lens_name] = np.stack([
            (group[rank].to_numpy() <= k).astype(float) for k in ks
        ], axis=1)
        curves[lens_name] = (
            pd.DataFrame(curves[lens_name], index=group["item"].to_numpy())
            .groupby(level=0).mean()
        )
    items = curves[LENS_ORDER[0]].index.intersection(curves[LENS_ORDER[1]].index)
    diff = (curves[LENS_ORDER[0]].loc[items] - curves[LENS_ORDER[1]].loc[items]).to_numpy()
    per_item = ((diff[:, :-1] + diff[:, 1:]) / 2) @ weights
    rng = np.random.default_rng(seed)
    boots = per_item[rng.integers(0, len(per_item), (n_boot, len(per_item)))].mean(1)
    return float(per_item.mean()), *np.quantile(boots, [0.025, 0.975]).tolist()
