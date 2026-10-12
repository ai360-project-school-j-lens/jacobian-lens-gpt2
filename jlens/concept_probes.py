"""Batched concept activations, mean-difference probes, and ridge linear probes.

The template corpus explicitly names concepts. Held-out templates measure lexical
transfer across contexts, not decoding an unspoken concept. Latent transfer is a
separate causal evaluation in the concept-swap notebook.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Formatter

import pandas as pd
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel
from jlens.swap_experiment import ConceptDirections


def audit_probe_templates(templates: Sequence[str], *, seed: int = 0,
                          held_out: int = 5) -> pd.DataFrame:
    """Validate slots and split distinct causal prefixes into train/test contexts.

    Text after the concept cannot affect its activation in a causal decoder.
    Templates differing only in that suffix must never cross the split.
    """
    if not 1 <= held_out < len(templates):
        raise ValueError("Need nonempty training and held-out template sets")
    rows = []
    prefixes = set()
    for i, template in enumerate(templates):
        fields = [(field, spec, conversion) for _, field, spec, conversion in Formatter().parse(template)
                  if field is not None]
        if fields != [("x", "", None)]:
            raise ValueError("Each template must contain exactly one plain {x} slot")
        prefix, suffix = template.split("{x}")
        if prefix in prefixes:
            raise ValueError("Duplicate causal prefix: suffix-only changes are not independent contexts")
        prefixes.add(prefix)
        rows.append(dict(template_id=i, template=template, prefix=prefix, suffix=suffix))
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    test_indices = set(order[:held_out])
    for row in rows:
        row["split"] = "test" if row["template_id"] in test_indices else "train"
    return pd.DataFrame(rows)


def prepare_probe_examples(model: LensModel, concepts: Sequence[str], templates: pd.DataFrame, *,
                           max_length: int = 128) -> pd.DataFrame:
    """Locate each complete concept mention via tokenizer character offsets.

    The last token of the mention is the readout, including for multi-token
    concepts. Offset IDs must match the adapter's special-token policy exactly,
    including the BOS policy of an ``ExplicitPromptModel``.
    Unsupported offsets, merged boundaries, unknown tokens, and truncation are
    reported as exclusions; none uses a guessed token position.
    """
    tokenizer = model.tokenizer
    if len(set(concepts)) != len(concepts) or any(not concept.strip() for concept in concepts):
        raise ValueError("Concept labels must be nonempty and unique")
    rows = []
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    # An ExplicitPromptModel's BOS policy decides the special tokens, as in encode_ids.
    bos_policy = getattr(model, "bos_policy", None)
    special_tokens = {} if bos_policy is None else {"add_special_tokens": bos_policy == "tokenizer"}
    for concept in concepts:
        for template in templates.to_dict("records"):
            text = template["template"].format(x=concept)
            start = len(template["prefix"])
            end = start + len(concept)
            reason, position, span_tokens, ids = "", -1, 0, []
            try:
                encoded = tokenizer(text, return_offsets_mapping=True, truncation=True,
                                    max_length=max_length + 1, **special_tokens)
                offsets = list(encoded["offset_mapping"])
                ids = list(encoded["input_ids"])
                if bos_policy == "prepend":
                    ids.insert(0, tokenizer.bos_token_id)
                    offsets.insert(0, (0, 0))
                encode_ids = getattr(model, "encode_ids", None)
                canonical = (encode_ids(text, max_length=max_length + 1) if encode_ids else
                             model.encode(text, max_length=max_length + 1)[0].tolist())
                if ids != canonical:
                    reason = "offset encoding differs from model encoding"
                elif len(ids) > max_length:
                    reason = "over-length prompt"
                else:
                    overlap = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
                    if not overlap or offsets[overlap[0]][0] > start or offsets[overlap[-1]][1] != end:
                        reason = "incomplete or merged concept boundary"
                    elif any(ids[i] in special for i in overlap):
                        reason = "unknown or special token in concept"
                    else:
                        position, span_tokens = overlap[-1], len(overlap)
            except (NotImplementedError, TypeError, KeyError) as error:
                reason = f"offset mapping unavailable ({type(error).__name__})"
            rows.append(dict(concept=concept, **template, text=text, input_ids=ids,
                             position=position, concept_tokens=span_tokens,
                             valid=not reason, reason=reason))
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_probe_features(model: LensModel, examples: pd.DataFrame, layers: Sequence[int], *,
                           batch_size: int = 32, progress: bool = True) -> dict[int, torch.Tensor]:
    """Capture mention-end residuals with batched forwards; return CPU fp32 arrays."""
    if batch_size < 1 or examples.empty or not examples.valid.all():
        raise ValueError("Need positive batch size and nonempty, fully valid examples")
    layers = sorted(set(layers))
    if not layers or any(layer < 0 or layer >= model.n_layers for layer in layers):
        raise ValueError("Invalid probe layers")
    padded = bool(getattr(model, "supports_attention_mask", False))
    device = getattr(model, "input_device", None)
    if device is None:
        device = model.encode(examples.text.iloc[0]).device
    pad = getattr(model.tokenizer, "pad_token_id", None)
    pad = 0 if pad is None else pad
    rows = examples.reset_index(drop=True).to_dict("records")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[0 if padded else len(row["input_ids"])].append(i)
    batches = []
    for group in groups.values():
        group.sort(key=lambda i: len(rows[i]["input_ids"]))
        batches.extend(group[i:i + batch_size] for i in range(0, len(group), batch_size))
    features = {layer: torch.empty(len(rows), model.d_model) for layer in layers}
    for indices in tqdm(batches, desc="probe contexts", disable=not progress):
        lengths = [len(rows[i]["input_ids"]) for i in indices]
        width = max(lengths)
        ids = torch.tensor([rows[i]["input_ids"] + [pad] * (width - len(rows[i]["input_ids"]))
                            for i in indices], device=device)
        positions = torch.tensor([rows[i]["position"] for i in indices], device=device)
        batch_rows = torch.arange(len(indices), device=device)
        with ActivationRecorder(model.layers, at=layers) as recorder:
            if padded:
                mask = torch.arange(width, device=device)[None] < torch.tensor(lengths, device=device)[:, None]
                model.forward(ids, attention_mask=mask.long())
            else:
                model.forward(ids)
        for layer in layers:
            h = recorder.activations[layer]
            features[layer][indices] = h[batch_rows.to(h.device), positions.to(h.device)].float().cpu()
    return features


@torch.no_grad()
def fit_concept_probes(features: Mapping[int, torch.Tensor], examples: pd.DataFrame, *,
                       ridge: float = 0.1, device: str | torch.device = "cpu") -> dict:
    """Fit mean-difference and ridge classifiers on training templates only.

    Ridge minimizes mean squared one-hot error plus ``ridge * ||W||²`` after
    centering and one global RMS rescaling of training features. We map weights
    back into original residual coordinates. No held-out label selects lambda.
    Biases are used for classification diagnostics, not as intervention vectors.
    The primal/dual solve uses the smaller of hidden width and training examples.
    """
    if ridge <= 0 or not torch.isfinite(torch.tensor(ridge)):
        raise ValueError("ridge must be positive and finite")
    if not features or examples.empty or not examples.valid.all():
        raise ValueError("Need valid examples and features")
    concepts = tuple(sorted(examples.concept.unique()))
    if len(concepts) < 3:
        raise ValueError("At least three concepts are needed: two centered means are collinear")
    index = {concept: i for i, concept in enumerate(concepts)}
    labels = torch.tensor([index[c] for c in examples.concept], device=device)
    train = torch.tensor(examples.split.eq("train").tolist(), device=device)
    test = torch.tensor(examples.split.eq("test").tolist(), device=device)
    if not train.any() or not test.any() or not (train | test).all():
        raise ValueError("Need train and test examples with valid split labels")
    if set(examples.loc[examples.split.eq("train"), "prefix"]) & set(examples.loc[examples.split.eq("test"), "prefix"]):
        raise ValueError("Training and held-out causal prefixes overlap")
    counts = torch.bincount(labels[train], minlength=len(concepts))
    if not (counts > 0).all() or not (torch.bincount(labels[test], minlength=len(concepts)) > 0).all():
        raise ValueError("Every concept needs train and test coverage")
    targets = F.one_hot(labels[train], len(concepts)).float()
    target_mean = targets.mean(0)
    direction_sets = {name: {} for name in ("probe_mean", "probe_ridge")}
    biases = {name: {} for name in direction_sets}
    diagnostics = []
    train_count = int(examples.split.eq("train").sum())
    test_count = int(examples.split.eq("test").sum())
    for layer, raw in sorted(features.items()):
        if raw.ndim != 2 or raw.shape[0] != len(examples) or not torch.isfinite(raw).all():
            raise ValueError("Invalid probe feature shape or nonfinite activations")
        x = raw.to(device=device, dtype=torch.float32)
        training = x[train]
        mean = training.mean(0)
        means = torch.zeros(len(concepts), x.shape[1], device=device)
        means.index_add_(0, labels[train], training)
        means /= counts[:, None]
        mean_vectors = means - means.mean(0, keepdim=True)
        mean_bias = -0.5 * means.square().sum(-1)
        scale = (training - mean).square().mean().sqrt().clamp_min(1e-8)
        centered = (training - mean) / scale
        y = targets - target_mean
        n, d = centered.shape
        if n <= d:
            gram = centered @ centered.T
            gram.diagonal().add_(n * ridge)
            weights = centered.T @ torch.linalg.solve(gram, y)
        else:
            gram = centered.T @ centered
            gram.diagonal().add_(n * ridge)
            weights = torch.linalg.solve(gram, centered.T @ y)
        weights /= scale
        ridge_vectors = weights.T
        ridge_bias = target_mean - mean @ weights
        for name, vectors, bias in (("probe_mean", mean_vectors, mean_bias),
                                     ("probe_ridge", ridge_vectors, ridge_bias)):
            direction_sets[name][layer] = vectors.cpu()
            biases[name][layer] = bias.cpu()
            predicted = (x @ vectors.T + bias).argmax(-1)
            accuracies = torch.stack([(predicted[train] == labels[train]).float().mean(),
                                      (predicted[test] == labels[test]).float().mean()]).cpu().tolist()
            diagnostics.append(dict(layer=layer, method=name, train_accuracy=accuracies[0],
                                    test_accuracy=accuracies[1], chance=1 / len(concepts),
                                    train_examples=train_count, test_examples=test_count))
    return dict(concepts=list(concepts), directions=direction_sets, biases=biases,
                diagnostics=diagnostics, ridge=ridge)


def probe_directions(fitted: dict) -> dict[str, ConceptDirections]:
    """View fitted classifier weights as labeled residual intervention directions."""
    return {name: ConceptDirections(tuple(fitted["concepts"]), matrices)
            for name, matrices in fitted["directions"].items()}


def probe_fingerprint(metadata: dict, examples: pd.DataFrame, layers: Sequence[int]) -> str:
    """Key feature caches by runtime, exact prompts, token spans, layers and code."""
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(examples.to_json(orient="records").encode())
    digest.update(json.dumps(list(layers)).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:20]
