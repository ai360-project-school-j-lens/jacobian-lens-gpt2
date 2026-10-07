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
from jlens.vis import SliceData, build_page, notebook_iframe

from jlens.protocol import LensModel

Lens = Callable[[str], torch.Tensor]

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
        if message.startswith("  prompt "):
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


ActivationReadout = Callable[[LensModel, dict[int, torch.Tensor]], torch.Tensor]


@torch.no_grad()
def logit_lens_from_activations(
    model: LensModel, activations: dict[int, torch.Tensor]
) -> torch.Tensor:
    """Handwritten baseline: unembed each pre-final-norm block output."""
    logits = [
        model.unembed(activations[layer][0].float()).float()
        for layer in range(model.n_layers)
    ]
    return torch.stack(logits)


@torch.no_grad()
def logit_lens(model: LensModel, text: str) -> torch.Tensor:
    """Record one forward pass and read it out without Jacobian transport."""
    with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
        model.forward(model.encode(text))
    return logit_lens_from_activations(model, recorder.activations)


@torch.no_grad()
def jacobian_lens(model: LensModel, lens: JacobianLens, text: str) -> torch.Tensor:
    """Transport block outputs, then use the model's final norm and unembedding."""
    # активации снимаются теми же хуками на model.layers, что и при fit,
    # поэтому J_l применяется ровно к тому, на чём он обучен
    with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
        model.forward(model.encode(text))
    logits = []
    for layer in range(model.n_layers):
        residual = recorder.activations[layer][0].float()
        # у последнего блока J = I — это логиты модели
        if layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits.append(model.unembed(residual))
    return torch.stack(logits).float()


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
    logits = lens(prompt)
    layers = shown_layers(logits.shape[0], layer_stride)
    logits = logits[layers].transpose(0, 1).contiguous().cpu()  # [seq_len, n_layers, vocab]
    seq_len, n_layers, vocab_size = logits.shape

    top_ids = logits.topk(top_n, dim=-1).indices
    tracked = sorted(set(top_ids.flatten().tolist()))
    # ранг токена = число токенов словаря со строго большим логитом
    sorted_logits = logits.sort(dim=-1).values
    rank_tensor = vocab_size - torch.searchsorted(sorted_logits, logits[..., tracked], right=True)

    decode = lambda t: model.tokenizer.decode([t], clean_up_tokenization_spaces=False)
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
        top = lens(prompt)[layers, position].topk(top_n, dim=-1).indices.tolist()
        columns[name] = [[model.tokenizer.decode([t]) for t in row] for row in top]
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
    """Single-token spellings of `word`: with/without leading space, as-is/lower/capitalized."""
    ids = set()
    for spelling in synonyms(word) if expand else [word]:
        for variant in {spelling, spelling.lower(), spelling.capitalize()}:
            for text in (variant, " " + variant):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) == 1:
                    ids.add(tokens[0])
    return ids


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
    """1-based rank of every token of `token_ids` in `logits` [..., vocab] -> [..., len(token_ids)]."""
    ids = torch.tensor(sorted(token_ids), device=logits.device)
    values = logits[..., ids]
    return (logits.unsqueeze(-2) > values.unsqueeze(-1)).sum(-1) + 1


# --------------------------------------------------------------------------- #
# Readout evaluation (data/evaluations protocol)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_readout(model: LensModel, lens: Lens, dataset: str, items: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run `lens` over the dataset prompts and read it out at the readout position.

    Returns ``(words, items)``:

    * ``words`` — a row per (item, word). ``kind`` is ``intermediate``, ``target`` or
      ``control`` (intermediates of another item of the same dataset — a chance baseline).
      ``ranks`` is the 1-based rank at every layer, min over spellings (for order-ops —
      over the synonym set).
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
    logits: torch.Tensor,
    dataset: str,
    items: list[dict],
    i: int,
    ids: list[int],
) -> tuple[list[dict], dict]:
    """The shared evaluation protocol, independent of how logits were obtained."""
    tok = model.tokenizer
    expand = dataset == "order-ops"
    item = items[i]
    prompt = item["prompt"].rstrip()
    word_rows = []
    position = readout_position(tok, ids, dataset)
    readout = logits[:, position]  # [n_layers, vocab]

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
    for kind, word, role in words:
        word_ids = spelling_ids(tok, word, expand)
        ranks = token_ranks(readout, word_ids).min(-1).values.cpu().numpy()
        word_rows.append({
            "dataset": dataset,
            "item": item["name"],
            "kind": kind,
            "word": word,
            "role": role,
            "single_token": bool(single_token_ids(tok, word, expand)),
            "in_prompt": bool(word_ids & prompt_ids),
            "ranks": ranks,
            "best_rank": int(ranks.min()),
            "best_layer": int(ranks.argmin()),
        })

    # A tokenizer may emit BOS, multiple specials, or no specials at all.
    top1 = logits[:, positions].argmax(-1)
    log_probs = logits[:, positions].log_softmax(-1)
    kl = (log_probs[-1].exp() * (log_probs[-1] - log_probs)).sum(-1).mean(-1)
    input_ids = torch.tensor([ids[position] for position in positions], device=top1.device)
    model_top1 = int(readout[-1].argmax())
    item_row = {
        "dataset": dataset,
        "item": item["name"],
        "prompt": prompt,
        "target": item.get("target"),
        "readout_token": tok.decode([ids[position]]),
        "model_top1": tok.decode([model_top1]),
        "model_correct": model_top1 in spelling_ids(tok, item["target"]) if "target" in item else None,
        "readout_top1": [tok.decode([t]) for t in readout.argmax(-1).tolist()],
        "agreement": (top1 == top1[-1]).float().mean(-1).cpu().numpy(),
        "copy_rate": (top1 == input_ids).float().mean(-1).cpu().numpy(),
        "kl_to_final": kl.cpu().numpy(),
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
    logit_readout: ActivationReadout = logit_lens_from_activations,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate logit lens and J-lens from one set of block outputs per prompt.

    Returns the same metrics as :func:`evaluate_all`, with a ``lens`` column
    (``logit lens`` or ``J-lens``). Both use the exact same pre-final-norm residuals
    and final model logits. Only the J-lens transports inner-layer residuals.
    ``logit_readout(model, activations)`` can supply an explicit handwritten
    baseline, returning ``[n_layers, seq_len, vocab]`` logits. Its final row
    is set to the shared model output.
    All inner layers must be fitted; silently substituting the logit lens for
    a missing Jacobian would make the comparison misleading.

    Activations are released after each prompt, and only one lens's full
    logits tensor is kept at a time; dataset-wide vocabulary logits are not
    cached.
    """
    if lens.d_model != model.d_model:
        raise ValueError("lens d_model does not match the model")
    missing = set(range(model.n_layers - 1)) - set(lens.source_layers)
    if missing:
        raise ValueError(f"J-lens is missing inner layers {sorted(missing)}")

    word_rows, item_rows = [], []
    for dataset, samples in tqdm(evals.items(), desc=desc):
        for i, item in enumerate(tqdm(samples, desc=dataset, leave=False)):
            input_ids = model.encode(item["prompt"].rstrip())
            ids = input_ids[0].tolist()
            with ActivationRecorder(model.layers, at=range(model.n_layers)) as recorder:
                model.forward(input_ids)
            # The final raw block output goes through the adapter's complete
            # readout, including any model-specific logit softcap.
            final_logits = model.unembed(recorder.activations[model.n_layers - 1][0].float()).float()
            for name in ("logit lens", "J-lens"):
                if name == "logit lens":
                    logits = logit_readout(model, recorder.activations)
                    if logits.shape != (model.n_layers, *final_logits.shape):
                        raise ValueError("logit_readout must return [n_layers, seq_len, vocab]")
                    logits[-1] = final_logits
                else:
                    layer_logits = []
                    for layer in range(model.n_layers - 1):
                        residual = recorder.activations[layer][0].float()
                        transported = lens.transport(residual, layer)
                        layer_logits.append(model.unembed(transported).float())
                        del residual, transported
                    logits = torch.stack([*layer_logits, final_logits])
                    del layer_logits
                rows, row = _readout_rows(model, logits, dataset, samples, i, ids)
                word_rows.extend({**entry, "lens": name} for entry in rows)
                item_rows.append({**row, "lens": name})
                del logits
            del recorder, final_logits
    return pd.DataFrame(word_rows), pd.DataFrame(item_rows)


def _layers(n_layers: int, layers: slice | Sequence[int] | None) -> np.ndarray:
    return np.arange(n_layers)[slice(None) if layers is None else layers]


def pass_at_k(words: pd.DataFrame, ks: Sequence[int] = (1, 5, 10, 100), layers: slice | Sequence[int] | None = None) -> pd.DataFrame:
    """pass@k as in data/evaluations: mean over items of the fraction of words whose min-over-layers rank <= k.

    `layers` restricts the min to some layers, e.g. ``slice(None, -1)`` — без выхода модели.
    Long frame: [lens], dataset, kind, k, score.
    """
    keys = [c for c in ("lens", "dataset", "kind") if c in words]
    rows = []
    for key, group in words.groupby(keys):
        ranks = np.stack(group["ranks"])
        best = ranks[:, _layers(ranks.shape[1], layers)].min(1)
        for k in ks:
            score = pd.Series(best <= k, index=group.index).groupby(group["item"]).mean().mean()
            rows.append({**dict(zip(keys, key)), "k": k, "score": score})
    return pd.DataFrame(rows)


def layer_hit_rate(words: pd.DataFrame, k: int) -> pd.DataFrame:
    """Share of words with rank <= k at every layer: [dataset x layer]."""
    return pd.DataFrame({d: (np.stack(g["ranks"]) <= k).mean(0) for d, g in words.groupby("dataset", sort=False)}).T


def layer_median_rank(words: pd.DataFrame) -> pd.DataFrame:
    """Median rank at every layer: [dataset x layer]."""
    return pd.DataFrame({d: np.median(np.stack(g["ranks"]), 0) for d, g in words.groupby("dataset", sort=False)}).T


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
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), sharex=True, squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        for label, frame in curves.items():
            if dataset in frame.index:
                ax.plot(frame.columns, frame.loc[dataset], marker="o", ms=3, label=label,
                        linestyle="--" if "control" in label else "-")
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
