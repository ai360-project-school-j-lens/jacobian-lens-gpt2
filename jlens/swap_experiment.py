"""Audited datasets and batched causal swaps for model-independent notebooks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from jlens.evaluation import single_token_ids
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.workspace import (
    _jacobian_on,
    _module,
    final_norm_centers,
    final_norm_gain,
)


@dataclass
class ConceptDirections:
    """Per-layer residual directions, ordered by concept label, with no token fallback."""

    concepts: tuple[str, ...]
    vectors: dict[int, torch.Tensor]


def load_swap_trials(root: str | Path) -> pd.DataFrame:
    """Load original Anthropic sets and local v1, retaining every exclusion."""
    directory = Path(root) / "data/experiments"
    audit = json.loads((directory / "token-swap-audit.json").read_text())
    rows = []
    for dataset in ("probe-swap", "latent-bridge"):
        data = json.loads((directory / f"{dataset}.json").read_text())
        for item in data["items"]:
            reasons = []
            if dataset == "probe-swap":
                reasons.append(audit["probe_exclusions"].get(item["name"], ""))
            else:
                reasons.append(audit["latent_landmark_exclusions"].get(item["landmark"], ""))
                for key in ("intermediate", "swap_to"):
                    reasons.append(audit["latent_concept_exclusions"].get(item[key], ""))
            rows.append(dict(
                dataset=dataset, name=item["name"], group=item["category"],
                prompt=item["prompt"].rstrip(), source=item["intermediate"],
                target=item["swap_to"], answer=str(item["answer"]),
                swap_answer=str(item["swap_answer"]),
                quality_reason="; ".join(dict.fromkeys(r for r in reasons if r)),
            ))
    data = json.loads((directory / "flexible-generalization.json").read_text())
    for category in data["categories"]:
        for function in category["funcs"]:
            group = f"{category['name']}/{function['name']}"
            for source in category["args"]:
                for target in category["args"]:
                    if source == target:
                        continue
                    rows.append(dict(
                        dataset="flexible-generalization", group=group,
                        name=f"{group}:{source}->{target}",
                        prompt=function["template"].format(arg=source).rstrip(),
                        source=source, target=target,
                        answer=str(function["answers"][source]),
                        swap_answer=str(function["answers"][target]),
                        quality_reason=audit["flexible_function_exclusions"].get(group, ""),
                    ))
    frame = pd.DataFrame(rows)
    frame["quality_ok"] = frame.quality_reason.eq("")
    frame["trial_id"] = frame.dataset + "/" + frame.name
    if frame.trial_id.duplicated().any():
        raise ValueError("Duplicate trial identifiers")
    return frame


def prepare_swap_trials(model: LensModel, trials: pd.DataFrame, *,
                        max_length: int = 256,
                        require_single_token_concepts: bool = True) -> pd.DataFrame:
    """Audit complete tokens, disjoint answers, and prompt truncation per model.

    Concepts prefer the exact leading-space spelling, then exact bare spelling.
    Answers accept complete case/space variants; fragments are never accepted.
    Prompt encoding uses the adapter's own special-token policy. Set
    ``require_single_token_concepts=False`` for learned-concept-only cohorts;
    answer coverage and disjointness still apply.
    """
    tokenizer = model.tokenizer
    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            specials.add(value)

    def concept(word):
        for spelling in (" " + word, word):
            ids = tokenizer.encode(spelling, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in specials and tokenizer.decode(
                ids, clean_up_tokenization_spaces=False,
            ) == spelling:
                return ids[0]
        return None

    concepts = {word: concept(word) for word in set(trials.source) | set(trials.target)}
    answers = {word: sorted(single_token_ids(tokenizer, word))
               for word in set(trials.answer) | set(trials.swap_answer)}
    prompts = {}
    for prompt in trials.prompt.unique():
        encode_ids = getattr(model, "encode_ids", None)
        ids = (encode_ids(prompt, max_length=max_length + 1) if encode_ids else
               model.encode(prompt, max_length=max_length + 1)[0].tolist())
        prompts[prompt] = ids
    rows = []
    for row in trials.to_dict("records"):
        reasons = []
        source, target = concepts[row["source"]], concepts[row["target"]]
        original, swapped = answers[row["answer"]], answers[row["swap_answer"]]
        ids = prompts[row["prompt"]]
        if require_single_token_concepts and (source is None or target is None):
            reasons.append("multi-token or invalid concept")
        elif require_single_token_concepts and source == target:
            reasons.append("identical concept tokens")
        if not original or not swapped:
            reasons.append("answer has no complete single-token spelling")
        elif set(original) & set(swapped):
            reasons.append("overlapping answer tokens")
        if not ids or len(ids) > max_length:
            reasons.append("empty or over-length prompt")
        rows.append(dict(row, source_id=source, target_id=target,
                         answer_ids=original, swap_answer_ids=swapped,
                         input_ids=ids, token_reason="; ".join(reasons),
                         token_ok=not reasons, eligible=row["quality_ok"] and not reasons))
    return pd.DataFrame(rows)


def intermediate_bands(n_layers: int) -> dict[str, list[int]]:
    """Three predeclared middle-layer bands, excluding the final block."""
    if n_layers < 3:
        raise ValueError("At least three blocks are needed for intermediate swaps")
    bands = {}
    for lo, hi in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8)):
        start = max(0, min(n_layers - 2, round(lo * n_layers)))
        stop = min(n_layers - 1, max(start + 1, round(hi * n_layers)))
        bands[f"{start}:{stop}"] = list(range(start, stop))
    return bands


class BatchedSwap:
    """Clamp independent per-row concept coordinates at all selected positions.

    ``bases[layer]`` is [batch, width, 2]; clean activations are [batch, seq, width].
    Random controls replay the actual J arm's per-position update norms.
    Hooks preserve tuple outputs and are always removed, including on exceptions.
    """

    def __init__(self, model: LensModel, bases: Mapping[int, torch.Tensor],
                 clean: Mapping[int, torch.Tensor], positions: torch.Tensor, *,
                 alpha: float = 1.0, random_norms: Mapping[int, torch.Tensor] | None = None,
                 seed: int = 0) -> None:
        self.model, self.bases, self.clean = model, bases, clean
        self.positions, self.alpha = positions, alpha
        self.random_norms = random_norms
        self.generator = torch.Generator().manual_seed(seed)
        self.inverses = {layer: torch.linalg.pinv(basis) for layer, basis in bases.items()}
        self.update_norms: dict[int, torch.Tensor] = {}
        self.handles = []

    def _hook(self, layer):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            if self.alpha == 0:
                return output
            basis = self.bases[layer].to(h.device)
            inverse = self.inverses[layer].to(h.device)
            coords = self.clean[layer].to(h.device).float() @ inverse.transpose(1, 2)
            wanted = coords + self.alpha * (coords.flip(-1) - coords)
            current = h.float() @ inverse.transpose(1, 2)
            update = (wanted - current) @ basis.transpose(1, 2)
            if self.random_norms is not None:
                noise = torch.randn(update.shape, generator=self.generator).to(h.device)
                update = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                update = update * self.random_norms[layer].to(h.device)
            update = update * self.positions.to(h.device).unsqueeze(-1)
            self.update_norms[layer] = update.norm(dim=-1, keepdim=True)
            patched = h + update.to(h.dtype)
            return patched if torch.is_tensor(output) else (patched, *output[1:])
        return hook

    def __enter__(self):
        try:
            for layer in self.bases:
                self.handles.append(self.model.layers[layer].register_forward_hook(self._hook(layer)))
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _score(logits, answer_ids):
    width = max(map(len, answer_ids))
    ids = torch.tensor([ids + [ids[0]] * (width - len(ids)) for ids in answer_ids],
                       device=logits.device)
    mask = torch.tensor([[True] * len(ids) + [False] * (width - len(ids))
                         for ids in answer_ids], device=logits.device)
    values = logits.gather(1, ids).masked_fill(~mask, -torch.inf)
    best = values.max(dim=1).values
    top = logits.argmax(dim=-1)
    hit = ((top[:, None] == ids) & mask).any(dim=1)
    ranks = (logits > best[:, None]).sum(dim=-1) + 1
    # A tied top answer is not evidence of baseline capability or a causal flip.
    unique = (logits == logits.max(dim=-1, keepdim=True).values).sum(dim=-1) == 1
    probability = (logits.log_softmax(-1).gather(1, ids).exp() * mask).sum(-1)
    return top, hit & unique, ranks, probability


@torch.no_grad()
def run_swap_experiment(model: LensModel, lens: JacobianLens, trials: pd.DataFrame, *,
                        bands: Mapping[str, Sequence[int]] | None = None,
                        alphas: Sequence[float] = (0.0, 1.0, 2.0), batch_size: int = 16,
                        seed: int = 0, progress: bool = True,
                        concept_directions: Mapping[str, ConceptDirections] | None = None,
                        token_controls: bool = True) -> pd.DataFrame:
    """Batch prompts and compare token directions and/or learned concept directions.

    Mask-capable adapters use right padding; other protocols use exact-length
    batches. Final logits come from unembedding the last real token's final
    block output. BOS, EOS, and other special tokens are never patched.
    All eligible trials are retained; analysis conditions on clean correctness.
    Each learned arm gets its own ``random_<arm>`` norm-matched control.
    With ``token_controls=False``, token IDs are unused and multi-token concepts
    can be evaluated through their learned vectors alone.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    bands = dict(intermediate_bands(model.n_layers) if bands is None else bands)
    layers = sorted({layer for band in bands.values() for layer in band})
    if not layers or any(not band for band in bands.values()) or any(
        layer < 0 or layer >= model.n_layers - 1 for layer in layers
    ):
        raise ValueError("Bands must contain intermediate layers only")
    if lens.d_model != model.d_model or not set(layers).issubset(lens.source_layers):
        raise ValueError("Lens width or layer coverage does not match model")
    if any(not torch.isfinite(torch.tensor(alpha)) or alpha < 0 for alpha in alphas):
        raise ValueError("alphas must be finite and nonnegative")
    rows = trials[trials.eligible].to_dict("records")
    if not rows:
        return pd.DataFrame()
    head = _module(model, "_lm_head", "lm_head") if token_controls else None
    device = getattr(model, "input_device", None)
    if device is None:
        device = model.encode(rows[0]["prompt"]).device
    concept_directions = dict(concept_directions or {})
    if not token_controls and not concept_directions:
        raise ValueError("Need token controls or at least one learned concept arm")
    labels = sorted({row[key] for row in rows for key in ("source", "target")})
    label_index = {label: index for index, label in enumerate(labels)}
    vector_tables = {}
    if token_controls:
        tokens = sorted({int(row[key]) for row in rows for key in ("source_id", "target_id")})
        token_index = {token: index for index, token in enumerate(tokens)}
        directions = final_norm_gain(model) * head.weight[tokens].float()
        if final_norm_centers(model):
            directions -= directions.mean(dim=-1, keepdim=True)
        vector_tables["J"] = {
            layer: directions @ _jacobian_on(lens, layer, directions.device) for layer in layers
        }
        vector_tables["logit"] = dict.fromkeys(layers, directions)
    for name, supplied in concept_directions.items():
        if name in {"J", "logit", "random"} or name.startswith("random_"):
            raise ValueError(f"Reserved concept arm name: {name}")
        if len(set(supplied.concepts)) != len(supplied.concepts):
            raise ValueError(f"Duplicate concept labels in {name}")
        if not set(labels).issubset(supplied.concepts) or not set(layers).issubset(supplied.vectors):
            raise ValueError(f"Missing concept or layer coverage in {name}")
        selected = [supplied.concepts.index(label) for label in labels]
        vector_tables[name] = {}
        for layer in layers:
            matrix = supplied.vectors[layer]
            if matrix.shape != (len(supplied.concepts), model.d_model) or not torch.isfinite(matrix).all():
                raise ValueError(f"Invalid direction matrix for {name}, layer {layer}")
            vector_tables[name][layer] = matrix[selected].to(device=device, dtype=torch.float32)
    arms = [("J", None), ("logit", None), ("random", "J")] if token_controls else []
    for name in concept_directions:
        arms.extend([(name, None), (f"random_{name}", name)])
    padded = bool(getattr(model, "supports_attention_mask", False))
    groups = defaultdict(list)
    for row in rows:
        groups[0 if padded else len(row["input_ids"])].append(row)
    batches = []
    for group in groups.values():
        group.sort(key=lambda row: len(row["input_ids"]))
        batches.extend(group[i:i + batch_size] for i in range(0, len(group), batch_size))
    tokenizer = model.tokenizer
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if (value := getattr(tokenizer, attr, None)) is not None:
            special.add(value)
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = 0 if pad is None else pad
    output = []
    for batch_number, batch in enumerate(tqdm(batches, disable=not progress, desc="swap batches")):
        lengths = torch.tensor([len(row["input_ids"]) for row in batch], device=device)
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.tensor([row["input_ids"] + [pad] * (width - len(row["input_ids"]))
                            for row in batch], device=device)
        mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
        positions = mask & ~torch.isin(ids, torch.tensor(sorted(special), device=device, dtype=torch.long))
        indices = torch.arange(len(batch), device=device)

        def forward(ids=ids, mask=mask):
            if padded:
                model.forward(ids, attention_mask=mask.long())
            else:
                model.forward(ids)

        def logits_of(recorder, indices=indices, lengths=lengths):
            final = recorder.activations[model.n_layers - 1]
            return model.unembed(final[indices.to(final.device), (lengths - 1).to(final.device)]).float()

        with ActivationRecorder(model.layers, at=[*layers, model.n_layers - 1]) as recorder:
            forward()
            clean = {layer: recorder.activations[layer].detach() for layer in layers}
            baseline = logits_of(recorder)
        base_top, base_correct, _, _ = _score(baseline, [row["answer_ids"] for row in batch])
        _, _, base_rank, base_prob = _score(baseline, [row["swap_answer_ids"] for row in batch])
        label_pairs = torch.tensor([[label_index[row[key]] for key in ("source", "target")]
                                    for row in batch], device=device)
        if token_controls:
            token_pairs = torch.tensor([[token_index[int(row[key])] for key in ("source_id", "target_id")]
                                        for row in batch], device=head.weight.device)
        bases_by_arm = {}
        for mode, tables in vector_tables.items():
            pairs = token_pairs if mode in {"J", "logit"} else label_pairs
            bases_by_arm[mode] = {
                layer: tables[layer][pairs.to(tables[layer].device)].transpose(1, 2) for layer in layers
            }
            # A rank-deficient pair cannot carry two independently swapped coordinates.
            if any(bool((torch.linalg.matrix_rank(basis) < 2).any()) for basis in bases_by_arm[mode].values()):
                raise ValueError(f"Degenerate {mode} concept pair; cannot exchange two coordinates")
        for band_name, band in bands.items():
            for alpha in alphas:
                update_norms = {}
                for mode, random_reference in arms:
                    basis_mode = random_reference or mode
                    bases = {layer: bases_by_arm[basis_mode][layer] for layer in band}
                    with BatchedSwap(model, bases, clean, positions, alpha=alpha,
                                     random_norms=update_norms[random_reference] if random_reference and alpha else None,
                                     seed=seed + batch_number) as intervention:
                        with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as recorder:
                            forward()
                            logits = logits_of(recorder)
                    if random_reference is None:
                        update_norms[mode] = intervention.update_norms
                    top, success, rank, probability = _score(logits, [row["swap_answer_ids"] for row in batch])
                    changed = top != base_top
                    values = torch.stack([base_top, base_correct, base_rank, base_prob,
                                          top, success, rank, probability, changed], dim=-1).cpu().tolist()
                    for row, value in zip(batch, values, strict=True):
                        bt, bc, br, bp, top_id, hit, rank_value, prob, change = value
                        output.append({key: row[key] for key in (
                            "trial_id", "dataset", "group", "name", "prompt", "source", "target", "answer", "swap_answer"
                        )} | dict(band=band_name, alpha=alpha, mode=mode,
                                  base_top=int(bt), base_correct=bool(bc), base_rank=br,
                                  base_probability=bp, top=int(top_id), success=bool(hit),
                                  rank=rank_value, probability=prob, changed=bool(change),
                                  baseline_text=tokenizer.decode([int(bt)]),
                                  patched_text=tokenizer.decode([int(top_id)])))
    return pd.DataFrame(output)


def experiment_fingerprint(metadata: dict, prepared: pd.DataFrame, lens: JacobianLens, *,
                           concept_directions: Mapping[str, ConceptDirections] | None = None) -> str:
    """Hash runtime/config, exact tokenized trials, implementation and lens bytes."""
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    digest.update(prepared.to_json(orient="records").encode())
    for module in (Path(__file__), Path(__file__).with_name("workspace.py")):
        digest.update(module.read_bytes())
    for layer, matrix in sorted(lens.jacobians.items()):
        digest.update(str(layer).encode())
        digest.update(matrix.detach().cpu().contiguous().numpy().tobytes())
    for name, supplied in sorted((concept_directions or {}).items()):
        digest.update(json.dumps([name, supplied.concepts]).encode())
        for layer, matrix in sorted(supplied.vectors.items()):
            digest.update(str(layer).encode())
            digest.update(matrix.detach().cpu().float().contiguous().numpy().tobytes())
    return digest.hexdigest()[:20]
