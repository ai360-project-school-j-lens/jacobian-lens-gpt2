"""Batched, model-independent injected-concept report experiments.

One teacher-forced prompt is shared across trials. Only original prompt
positions are patched, including during fixed-length greedy continuations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.readout import LensReadout, readout, selected_token_ranks
from jlens.workspace import (
    InjectDirections,
    clear_jacobian_cache,
    final_norm_centers,
    lens_directions,
)

MODES = ("J", "logit", "random")


def prepare_introspection_prompt(
    tokenizer: Any,
    spec: Mapping,
    *,
    prefill_key: str = "word",
    prompt_style: str = "auto",
    bos_policy: str = "auto",
    max_length: int = 512,
) -> dict:
    """Render a dialogue and locate the final user turn with tokenizer offsets.

    ``auto`` uses the checkpoint's chat template when present, otherwise plain
    ``User:`` / ``Assistant:`` turns. Chat templates own their special tokens;
    plain text gets one BOS when available. Explicit BOS policies are ``none``,
    ``prepend`` and ``tokenizer``. A template's existing BOS is never duplicated.
    Truncation and an automatic trailing EOS are rejected: readout is at the quote.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast-tokenizer offsets are required for the injected span")
    if prompt_style not in {"auto", "plain", "chat"}:
        raise ValueError("prompt_style must be auto, plain, or chat")
    if bos_policy not in {"auto", "none", "prepend", "tokenizer"}:
        raise ValueError("bos_policy must be auto, none, prepend, or tokenizer")
    messages = spec["intro_prompt"]
    if (
        len(messages) < 2
        or messages[-1] != {"role": "assistant", "content": ""}
        or messages[-2]["role"] != "user"
    ):
        raise ValueError("intro_prompt must end with a user and empty assistant turn")
    prefill = spec["prefills"][prefill_key]
    if not prefill.endswith('"'):
        raise ValueError("The prefill must end at an open quote")
    if prompt_style == "auto":
        prompt_style = "chat" if getattr(tokenizer, "chat_template", None) else "plain"
    if prompt_style == "chat":
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        roles = {"user": "User:", "assistant": "Assistant:", "system": "System:"}
        prefix = "\n\n".join(roles[m["role"]] + m["content"] for m in messages[:-1])
        prefix += "\n\nAssistant:"
    text = prefix + prefill
    question = messages[-2]["content"].strip()
    start = prefix.rfind(question) if question else -1
    if start < 0:
        raise ValueError("The rendered template must preserve the user question text")
    end = start + len(question)
    if bos_policy == "auto":
        bos_policy = (
            "prepend" if prompt_style == "plain" and tokenizer.bos_token_id is not None
            else "none"
        )
    encoded = tokenizer(
        text, add_special_tokens=bos_policy == "tokenizer",
        return_offsets_mapping=True, truncation=False,
    )
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if bos_policy == "prepend":
        if tokenizer.bos_token_id is None:
            raise ValueError("prepend requires a BOS token ID")
        if not ids or ids[0] != tokenizer.bos_token_id:
            ids.insert(0, tokenizer.bos_token_id)
            offsets.insert(0, (0, 0))
    if not ids or len(ids) > max_length:
        raise ValueError(f"Prompt has {len(ids)} tokens; require 1..{max_length}, no truncation")
    if offsets[-1][1] != len(text) or ids[-1] in tokenizer.all_special_ids:
        raise ValueError("Readout must be at the quote, without a trailing special token")
    user_positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not user_positions or user_positions[-1] == len(ids) - 1:
        raise ValueError("The user turn must be nonempty and precede the readout")
    whole_start = int(ids[0] == tokenizer.bos_token_id)
    return dict(
        text=text, input_ids=ids, style=prompt_style, bos_policy=bos_policy,
        prefill_key=prefill_key, question_char_span=[start, end],
        spans={"user turn": user_positions, "whole prompt": list(range(whole_start, len(ids)))},
    )


def introspection_concepts(tokenizer: Any, spec: Mapping, prompt: Mapping) -> pd.DataFrame:
    """Audit exact single-token concepts in the actual open-quote context.

    Unsupported concepts remain in the returned table with an exclusion reason.
    Never substitute a first subtoken, a synonym, or a leading-space variant.
    Re-encoding must preserve the entire prefill and append exactly one token.
    """
    raw = tokenizer.encode(prompt["text"], add_special_tokens=False)
    concepts = spec["concepts"]
    surfaces = [c["surface"] for c in concepts]
    names = [c["name"] for c in concepts]
    if not concepts or len(set(names)) != len(names) or len(set(surfaces)) != len(surfaces):
        raise ValueError("Require nonempty concepts with unique names and surfaces")
    encoded = tokenizer(
        [prompt["text"] + surface for surface in surfaces],
        add_special_tokens=False, truncation=False,
    )["input_ids"]
    rows = []
    for concept, ids in zip(concepts, encoded, strict=True):
        suffix = ids[len(raw):]
        token_id = suffix[0] if len(suffix) == 1 else None
        if ids[:len(raw)] != raw:
            reason = "prefill boundary retokenized"
        elif len(suffix) != 1:
            reason = "not one continuation token"
        elif token_id in tokenizer.all_special_ids:
            reason = "special or unknown token"
        elif tokenizer.decode([token_id], clean_up_tokenization_spaces=False) != concept["surface"]:
            reason = "token does not decode to exact surface"
        else:
            reason = "supported"
        rows.append(dict(
            concept=concept["name"], surface=concept["surface"],
            origin=concept.get("origin", "original"), supported=reason == "supported",
            reason=reason, token_id=token_id if reason == "supported" else None,
        ))
    return pd.DataFrame(rows).astype({"token_id": "Int64"})


def introspection_bands(
    n_layers: int, fractions: Mapping[str, Sequence[float]],
) -> dict[str, list[int]]:
    """Resolve depth fractions into nonempty bands, excluding the final block."""
    if n_layers < 2 or not fractions:
        raise ValueError("Require at least two blocks and a nonempty band mapping")
    bands = {}
    for name, (lo, hi) in fractions.items():
        if not 0 <= lo < hi <= 1:
            raise ValueError("Band fractions must satisfy 0 <= lo < hi <= 1")
        start = min(round(lo * n_layers), n_layers - 2)
        stop = min(max(start + 1, round(hi * n_layers)), n_layers - 1)
        bands[name] = list(range(start, stop))
    return bands


@torch.no_grad()
def final_token_readout(model: LensModel, input_ids: torch.Tensor) -> LensReadout:
    """Unembed only the last token's final block output, applying final norm once.

    Inputs are equal-length, unpadded rows. The public readout contract retains
    model logit transforms (e.g. Gemma softcapping) and separate lexical scores.
    """
    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids)
    return readout(model, recorder.activations[final_layer][:, -1])


@torch.no_grad()
def injection_directions(
    model: LensModel,
    lens: JacobianLens,
    token_ids: Sequence[int],
    layers: Sequence[int],
    *,
    seed: int = 0,
) -> tuple[dict[int, torch.Tensor], bool]:
    """Build [J/logit/random, concept, channel] directions once for each layer.

    One CPU-seeded random vector per concept/layer is reused across all settings
    and continuations. Jacobians move to the device once; their temporary device
    cache is released once the smaller direction bank has been constructed.
    """
    layers = sorted(set(layers))
    if lens.d_model != model.d_model or not layers:
        raise ValueError("Need a matching lens width and nonempty layers")
    if any(layer < 0 or layer >= model.n_layers - 1 for layer in layers):
        raise ValueError("Injection bands must exclude the final block")
    if any(layer not in lens.jacobians for layer in layers):
        raise ValueError("The lens is missing an injection layer")
    if any(lens.jacobians[layer].shape != (model.d_model, model.d_model) for layer in layers):
        raise ValueError("The lens has an invalid Jacobian shape")
    centers = final_norm_centers(model)
    logit = lens_directions(model, None, token_ids, layers[0], mode="logit", centers=centers)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bank = {}
    try:
        for layer in layers:
            jacobian = lens_directions(model, lens, token_ids, layer, centers=centers)
            random = torch.randn(logit.shape, generator=generator).to(logit.device)
            bank[layer] = torch.stack([jacobian, logit, random])
    finally:
        clear_jacobian_cache()
    if any(not torch.isfinite(rows).all() or (rows.norm(dim=-1) == 0).any() for rows in bank.values()):
        raise ValueError("Injection directions must be finite and nonzero")
    return bank, centers


def _scores(result: LensReadout, targets: torch.Tensor) -> torch.Tensor:
    ranks = selected_token_ranks(result.logits, targets[:, None]).squeeze(-1)
    lexical = (
        ranks if result.logits is result.ranking_scores
        else selected_token_ranks(result.ranking_scores, targets[:, None]).squeeze(-1)
    )
    return torch.stack([ranks, lexical, result.logits.argmax(-1)], dim=-1)


@torch.no_grad()
def run_introspection(
    model: LensModel,
    lens: JacobianLens,
    prompt: Mapping,
    concepts: pd.DataFrame,
    bands: Mapping[str, Sequence[int]],
    strengths: Sequence[float],
    *,
    primary_band: str,
    batch_size: int = 32,
    seed: int = 0,
    demo_concepts: int = 3,
    demo_tokens: int = 10,
    progress: bool = False,
) -> dict:
    """Run paired injection arms, one shared clean pass, and batched examples.

    Each forward batches concepts, modes, and positive strengths at a fixed band
    and span. Trials share one prompt, so no padding or per-row readout search is
    needed. Norm scaling is per row/layer over the selected positions, including
    effects of earlier patched layers. Metrics use distribution-logit ranks with
    token-ID tie breaking; ``lexical_rank`` is a pre-softcap diagnostic.
    """
    strengths = list(strengths)
    if (
        batch_size < 1 or not strengths or len(set(strengths)) != len(strengths)
        or any(not np.isfinite(s) or s <= 0 for s in strengths)
        or concepts.empty or primary_band not in bands or demo_tokens < 0
        or demo_concepts < 0
    ):
        raise ValueError("Require concepts, positive unique strengths, valid batches and primary band")
    if concepts.concept.duplicated().any() or concepts.token_id.isna().any():
        raise ValueError("Concepts must be unique and have supported token IDs")
    length = len(prompt["input_ids"])
    for positions in prompt["spans"].values():
        if not positions or any(p < 0 or p >= length for p in positions):
            raise ValueError("Injected positions must be bounded within the original prompt")
    if not bands or any(not layers or len(set(layers)) != len(layers) for layers in bands.values()):
        raise ValueError("Every band needs distinct, nonempty layers")
    token_ids = concepts.token_id.astype(int).tolist()
    bank, centers = injection_directions(
        model, lens, token_ids, [layer for layers in bands.values() for layer in layers], seed=seed,
    )
    device = next(iter(bank.values())).device
    # Notebook models live on one device; the prompt and constant tensors move once.
    input_ids = torch.tensor([prompt["input_ids"]], device=device)
    targets = torch.tensor(token_ids, device=device)
    baseline = final_token_readout(model, input_ids)
    if not torch.isfinite(baseline.logits).all() or not torch.isfinite(baseline.ranking_scores).all():
        raise ValueError("Nonfinite clean readout")
    clean = _scores(LensReadout(
        baseline.logits.expand(len(concepts), -1),
        baseline.ranking_scores.expand(len(concepts), -1),
    ), targets).cpu().numpy()
    clean_top1 = int(clean[0, 2])
    vocab_size = baseline.logits.shape[-1]
    del baseline
    trials = pd.DataFrame(product(range(3), range(len(concepts)), strengths),
                          columns=["mode_idx", "concept_idx", "strength"])
    mode_idx = torch.tensor(trials.mode_idx.to_numpy(), device=device)
    concept_idx = torch.tensor(trials.concept_idx.to_numpy(), device=device)
    scales = torch.tensor(trials.strength.to_numpy(), device=device, dtype=torch.float32)

    def inject(layers, positions, modes, indices, values):
        return InjectDirections(
            model, layers, {layer: bank[layer][modes, indices] for layer in layers},
            values, positions=positions,
        )

    frames = []
    settings = list(product(prompt["spans"], bands))
    if progress:
        from tqdm.auto import tqdm

        settings = tqdm(settings, desc="Injection spans / bands")
    forward_count = 1
    for span, band in settings:
        pieces = []
        finite = torch.ones((), dtype=torch.bool, device=device)
        for start in range(0, len(trials), batch_size):
            stop = min(start + batch_size, len(trials))
            idx = concept_idx[start:stop]
            with inject(bands[band], prompt["spans"][span], mode_idx[start:stop], idx, scales[start:stop]):
                result = final_token_readout(model, input_ids.expand(stop - start, -1))
            finite &= torch.isfinite(result.logits).all() & torch.isfinite(result.ranking_scores).all()
            pieces.append(_scores(result, targets[idx]))
            forward_count += 1
        # Transfer compact trial outcomes once per setting, never per concept.
        values = torch.cat(pieces).cpu().numpy()
        if not bool(finite):
            raise ValueError(f"Nonfinite readout at {span}, {band}; no results saved")
        frame = trials.copy()
        frame[["rank", "lexical_rank", "top1_id"]] = values
        control = pd.DataFrame(product(range(3), range(len(concepts)), [0.0]),
                               columns=trials.columns)
        control[["rank", "lexical_rank", "top1_id"]] = np.tile(clean, (3, 1))
        frames.append(pd.concat([control, frame], ignore_index=True).assign(span=span, band=band))
    results = pd.concat(frames, ignore_index=True)
    results["mode"] = np.array(MODES)[results.mode_idx]
    for column in ["concept", "surface", "origin", "token_id"]:
        results[column] = concepts[column].to_numpy()[results.concept_idx]
    decoded = {i: model.tokenizer.decode([i], clean_up_tokenization_spaces=False)
               for i in results.top1_id.unique().tolist()}
    results["top1"] = results.top1_id.map(decoded)
    results = results.drop(columns=["mode_idx", "concept_idx"])
    if not results["rank"].eq(1).equals(results.top1_id.eq(results.token_id)):
        raise AssertionError("Rank-1 must agree with the actual greedy model token")

    demos = []
    if demo_tokens and demo_concepts:
        # Fixed first concepts, minimum/maximum strengths, and primary band.
        demo_settings = [(0, c, 0.0) for c in range(min(demo_concepts, len(concepts)))]
        demo_settings += list(product(
            range(3), range(min(demo_concepts, len(concepts))), sorted({min(strengths), max(strengths)}),
        ))
        grid = pd.DataFrame(demo_settings, columns=trials.columns)
        for span, positions in prompt["spans"].items():
            texts, generated = [], []
            for start in range(0, len(grid), batch_size):
                chunk = grid.iloc[start:start + batch_size]
                modes = torch.tensor(chunk.mode_idx.to_numpy(), device=device)
                indices = torch.tensor(chunk.concept_idx.to_numpy(), device=device)
                values = torch.tensor(chunk.strength.to_numpy(), device=device)
                ids = input_ids.expand(len(chunk), -1)
                finite = torch.ones((), dtype=torch.bool, device=device)
                with inject(bands[primary_band], positions, modes, indices, values):
                    for _ in range(demo_tokens):
                        result = final_token_readout(model, ids)
                        finite &= torch.isfinite(result.logits).all()
                        ids = torch.cat([ids, result.logits.argmax(-1, keepdim=True)], dim=1)
                        forward_count += 1
                continuations = ids[:, length:].cpu().tolist()
                if not bool(finite):
                    raise ValueError("Nonfinite greedy readout; no results saved")
                generated.extend(continuations)
                texts.extend(model.tokenizer.decode(row, clean_up_tokenization_spaces=False)
                             for row in continuations)
            demo = grid.assign(span=span, band=primary_band, continuation=texts, generated_ids=generated)
            demo["mode"] = np.array(MODES)[demo.mode_idx]
            demo["concept"] = concepts.concept.to_numpy()[demo.concept_idx]
            demos.extend(demo.drop(columns=["mode_idx", "concept_idx"]).to_dict("records"))
    return dict(
        trials=results.to_dict("records"), demos=demos,
        diagnostics=dict(
            final_norm_centers=centers, clean_top1_id=clean_top1, clean_top1=decoded[clean_top1],
            vocab_size=vocab_size, forward_passes=forward_count, n_concepts=len(concepts),
        ),
    )


def load_introspection_cache(path: str | Path, *, expected_manifest: dict | None = None) -> dict:
    """Read a self-contained JSON cache without any tokenizer/model/Hub access.

    Validate the complete paired grid and shared controls before reanalysis;
    never apply current configuration or replacement concepts to saved ranks.
    """
    payload = json.loads(Path(path).read_text())
    manifest = payload["manifest"]
    if manifest.get("schema") != "introspection-v1":
        raise ValueError("Unsupported introspection cache schema")
    # JSON object keys (e.g. config.id2label) are strings on disk.
    expected = (
        json.loads(json.dumps(expected_manifest, allow_nan=False))
        if expected_manifest is not None else None
    )
    if expected is not None and manifest != expected:
        raise ValueError("Cache manifest mismatch")
    results = pd.DataFrame(payload["trials"])
    columns = ["span", "band", "strength", "mode", "concept"]
    expected = set(product(
        manifest["prompt"]["spans"], manifest["bands"], [0.0, *manifest["strengths"]],
        MODES, manifest["selected_concepts"],
    ))
    if (
        results.empty or results.duplicated(columns).any()
        or set(results[columns].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Cache must contain the complete paired trial grid")
    ranks = results[["rank", "lexical_rank"]].to_numpy(dtype=float)
    if not np.isfinite(ranks).all() or (ranks < 1).any() or not (ranks == np.floor(ranks)).all():
        raise ValueError("Cached ranks must be finite positive integers")
    if (ranks > payload["diagnostics"]["vocab_size"]).any():
        raise ValueError("Cached rank exceeds vocabulary size")
    control = results[results.strength.eq(0)]
    if (control.groupby("concept")[["rank", "lexical_rank", "top1_id"]].nunique() > 1).any().any():
        raise ValueError("The clean control must be shared across all settings")
    if not results["rank"].eq(1).equals(results.top1_id.eq(results.token_id)):
        raise ValueError("Cached rank-1 and greedy token disagree")
    return payload
