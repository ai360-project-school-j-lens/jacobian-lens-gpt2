"""Model-independent layer structure measurements for workspace Figures 27–28."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class LayerGeometry:
    """Exact vocabulary-centered linear CKA and PCA dimension fractions."""

    layers: list[int]
    cka: np.ndarray
    dimensions: pd.DataFrame


def _layers(model, lens, layers):
    selected = sorted(set(layers) | {model.n_layers - 1})
    if lens.d_model != model.d_model or not selected:
        raise ValueError("Lens and model widths must match")
    if any(not isinstance(i, int) or not 0 <= i < model.n_layers for i in selected):
        raise ValueError("layers must be block indices in model range")
    if set(selected[:-1]) - set(lens.source_layers):
        raise ValueError("Lens is missing requested inner layers")
    return selected


@torch.no_grad()
def workspace_geometry(
    model: LensModel,
    lens: JacobianLens,
    unembedding_weight: torch.Tensor,
    *,
    layers: Sequence[int],
    variance_thresholds: Sequence[float] = (0.9, 0.95, 0.98, 0.99, 0.995),
    vocab_chunk_size: int = 4096,
    device: str | torch.device = "cpu",
    progress: bool = True,
) -> LayerGeometry:
    """Measure raw ``W_U @ J_l`` geometry; omit norm gain, bias and softcaps.

    All vocabulary rows participate, including special tokens, and are centered
    across vocabulary. With ``C = U_centered.T @ U_centered / vocab_size`` and
    ``B_l = sqrt(C) @ J_l``, ``G_l = B_l @ B_l.T`` has the same nonzero
    eigenvalues and cross-layer Frobenius products as the vocabulary Gram
    matrix, up to a common scale. This avoids vocabulary-squared allocations.

    Compute in float64 on CPU/CUDA; retain normalized Gram matrices on CPU in
    float32 (O(layers * d_model**2) memory). Final layer always uses identity.
    Undefined zero-variance geometry is NaN. Dimension fractions divide by
    d_model, not numerical rank. The explicit weight supports any LensModel
    with a linear output projection without inspecting architecture internals.
    """
    selected = _layers(model, lens, layers)
    weight = unembedding_weight.detach()
    d = model.d_model
    if weight.ndim != 2 or weight.shape[1] != d or weight.shape[0] < 2:
        raise ValueError("unembedding_weight must have shape [vocab >= 2, d_model]")
    if (
        vocab_chunk_size < 1
        or not variance_thresholds
        or any(not 0 < q <= 1 for q in variance_thresholds)
    ):
        raise ValueError("Require positive chunk size and thresholds in (0, 1]")
    device = torch.device(device)
    covariance = torch.zeros(d, d, dtype=torch.float64, device=device)
    mean = torch.zeros(d, dtype=torch.float64, device=device)
    for chunk in weight.split(vocab_chunk_size):
        mean += chunk.to(device=device, dtype=torch.float64).sum(0)
    mean /= len(weight)
    for chunk in weight.split(vocab_chunk_size):
        centered = chunk.to(device=device, dtype=torch.float64) - mean
        covariance += centered.T @ centered / len(weight)
    if not torch.isfinite(covariance).all():
        raise ValueError("Nonfinite unembedding covariance")
    values, vectors = torch.linalg.eigh(covariance)
    # A factor R with R.T @ R = C suffices; symmetric sqrt is unnecessary.
    root = values.clamp_min(0).sqrt()[:, None] * vectors.T
    features = torch.empty(len(selected), d * d, dtype=torch.float32, device="cpu")
    rows = []
    thresholds = torch.tensor(variance_thresholds, dtype=torch.float64, device=device)
    for index, layer in enumerate(
        tqdm(selected, desc="J-space geometry", disable=not progress)
    ):
        projected = (
            root
            if layer == model.n_layers - 1
            else (root @ lens.jacobians[layer].to(device=device, dtype=torch.float64))
        )
        gram = projected @ projected.T
        if not torch.isfinite(gram).all():
            raise ValueError(f"Nonfinite geometry at layer {layer}")
        spectrum = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0)
        total = spectrum.sum()
        if total > 0:
            cumulative = spectrum.cumsum(0) / total
            cumulative[-1] = 1.0
            fractions = (
                ((torch.searchsorted(cumulative, thresholds) + 1) / d).cpu().tolist()
            )
            features[index] = (gram / gram.norm()).flatten().float().cpu()
        else:
            fractions = [float("nan")] * len(thresholds)
            features[index].fill_(float("nan"))
        rows.extend(
            dict(layer=layer, variance=q, dimension_fraction=f)
            for q, f in zip(variance_thresholds, fractions, strict=True)
        )
    cka = (features @ features.T).clamp(0, 1).numpy()
    return LayerGeometry(selected, cka, pd.DataFrame(rows))


def top1_autocorrelation(
    sequences: Sequence[np.ndarray],
    lags: Sequence[int],
) -> pd.DataFrame:
    """Log repeat probability relative to an exact within-text shuffle null.

    Sequences retain original positions; -1 marks excluded tokens/padding.
    A valid pair must have two valid endpoints in the same text. The null
    permutes valid positions independently within each text, without replacement.
    Pool expected matches with the same pair weighting as observed matches.
    Add 1/2 to both match counts to keep zero-hit estimates finite; absent
    pairs remain NaN. This estimator choice is explicit, not paper source code.
    """
    if any(lag < 1 for lag in lags):
        raise ValueError("lags must be positive")
    rows = []
    for lag in lags:
        matches = pairs = 0
        expected = 0.0
        for tokens in sequences:
            tokens = np.asarray(tokens)
            valid_tokens = tokens[tokens >= 0]
            n = len(valid_tokens)
            if len(tokens) <= lag or n < 2:
                continue
            _, counts = np.unique(valid_tokens, return_counts=True)
            null = (counts * (counts - 1)).sum() / (n * (n - 1))
            left, right = tokens[:-lag], tokens[lag:]
            valid = (left >= 0) & (right >= 0)
            count = int(valid.sum())
            pairs += count
            matches += int(((left == right) & valid).sum())
            expected += count * null
        rows.append(
            dict(
                lag=lag,
                n_pairs=pairs,
                matches=matches,
                null_expected_matches=expected,
                delta_log_p=np.log((matches + 0.5) / (expected + 0.5))
                if pairs
                else np.nan,
            )
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def workspace_readouts(
    model: LensModel,
    lens: JacobianLens,
    texts: Sequence[str],
    *,
    layers: Sequence[int],
    ks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    percentiles: Sequence[float] = (1, 10, 25, 50, 75, 90, 99),
    lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
    max_seq_len: int = 128,
    batch_size: int = 8,
    position_chunk_size: int = 32,
    skip_first: int = 0,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Batched, token-weighted Figure 28 readouts from shared block outputs.

    Right-pad length-sorted batches. LensModel is causal, so right padding
    cannot affect real positions; padding is never scored. Exclude special
    input tokens (including BOS), retaining original positions for lags. All
    real positions, including each text's last one, have a model prediction.
    Distribution logits from model.unembed include model-specific transforms.
    Top-k ties use score descending then token ID ascending. Constant-logit
    kurtosis is undefined and is excluded with its coverage count reported.
    Store only top-1 IDs and scalar kurtoses, never full corpus logits.
    """
    selected = _layers(model, lens, layers)
    if min(batch_size, position_chunk_size, max_seq_len) < 1 or skip_first < 0:
        raise ValueError("Sizes must be positive and skip_first nonnegative")
    if (
        not ks
        or min(ks) < 1
        or not percentiles
        or any(not 0 <= q <= 100 for q in percentiles)
    ):
        raise ValueError("Require positive ks and percentiles in [0, 100]")
    encoded = [model.encode(t, max_length=max_seq_len)[0] for t in texts]
    encoded = sorted((ids for ids in encoded if len(ids)), key=len)
    if not encoded:
        raise ValueError("No tokens to evaluate")
    device = encoded[0].device
    final = model.n_layers - 1
    matrices = {
        i: lens.jacobians[i].to(device=device, dtype=torch.float32)
        for i in selected
        if i != final
    }
    special_ids = set(getattr(model.tokenizer, "all_special_ids", []))
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token = getattr(model.tokenizer, name, None)
        if token is not None:
            special_ids.add(token)
    special = torch.tensor(sorted(special_ids), device=device, dtype=torch.long)
    pad = getattr(model.tokenizer, "pad_token_id", None)
    pad = pad if pad is not None else int(encoded[0][0].cpu())
    hits = {i: torch.zeros(len(ks), dtype=torch.int64, device=device) for i in selected}
    k_tensor = torch.tensor(ks, device=device)
    all_finite = torch.ones((), dtype=torch.bool, device=device)
    kurtoses = {i: [] for i in selected}
    sequences = {i: [] for i in selected}
    n_tokens = 0
    for start in tqdm(
        range(0, len(encoded), batch_size),
        desc="Batched readouts",
        disable=not progress,
    ):
        batch = encoded[start : start + batch_size]
        lengths = torch.tensor([len(ids) for ids in batch], device=device)
        ids = torch.nn.utils.rnn.pad_sequence(
            batch, batch_first=True, padding_value=pad
        )
        positions = torch.arange(ids.shape[1], device=device)[None, :]
        valid = (positions < lengths[:, None]) & (positions >= skip_first)
        valid &= ~torch.isin(ids, special)
        indices = valid.flatten().nonzero().flatten()
        if not len(indices):
            continue
        n_tokens += len(indices)
        with ActivationRecorder(model.layers, at=selected) as recorder:
            model.forward(ids)
        activations = recorder.activations
        reference = activations[final].reshape(-1, model.d_model)[indices].float()
        final_top = torch.cat(
            [
                model.unembed(chunk).argmax(-1)
                for chunk in reference.split(position_chunk_size)
            ]
        )
        for layer in selected:
            residual = activations[layer].reshape(-1, model.d_model)[indices].float()
            top_ids, values = [], []
            for offset in range(0, len(residual), position_chunk_size):
                h = residual[offset : offset + position_chunk_size]
                if layer != final:
                    h = h @ matrices[layer].T
                logits = model.unembed(h).float()
                all_finite &= torch.isfinite(logits).all()
                # Rank only the reference token; avoids a full vocabulary sort.
                target = final_top[offset : offset + len(h)]
                score = logits.gather(1, target[:, None])
                token_ids = torch.arange(logits.shape[-1], device=device)[None, :]
                rank = (
                    (logits > score)
                    | ((logits == score) & (token_ids < target[:, None]))
                ).sum(-1) + 1
                hits[layer] += (rank[:, None] <= k_tensor).sum(0)
                top_ids.append(logits.argmax(-1))
                centered = logits - logits.mean(-1, keepdim=True)
                variance = centered.square().mean(-1)
                kurtosis = centered.pow(4).mean(-1) / variance.square() - 3
                values.append(kurtosis)
            dense_top = torch.full(ids.shape, -1, dtype=torch.long, device=device)
            dense_top.view(-1)[indices] = torch.cat(top_ids)
            sequences[layer].extend(dense_top.cpu().numpy())
            kurtoses[layer].append(torch.cat(values).cpu().numpy())
        del activations, recorder, reference
    if not n_tokens:
        raise ValueError("No nonspecial positions remain")
    if not all_finite:
        raise ValueError("Nonfinite readout logits")
    accuracy, kurtosis_rows, autocorrelation = [], [], []
    for layer in selected:
        counts = hits[layer].cpu().tolist()
        accuracy.extend(
            dict(layer=layer, k=k, accuracy=count / n_tokens, n_tokens=n_tokens)
            for k, count in zip(ks, counts, strict=True)
        )
        values = np.concatenate(kurtoses[layer])
        finite = values[np.isfinite(values)]
        quantiles = (
            np.percentile(finite, percentiles)
            if len(finite)
            else [np.nan] * len(percentiles)
        )
        kurtosis_rows.extend(
            dict(
                layer=layer,
                percentile=q,
                excess_kurtosis=v,
                n_valid=len(finite),
                n_tokens=n_tokens,
            )
            for q, v in zip(percentiles, quantiles, strict=True)
        )
        autocorrelation.append(
            top1_autocorrelation(sequences[layer], lags).assign(layer=layer)
        )
    return dict(
        accuracy=pd.DataFrame(accuracy),
        kurtosis=pd.DataFrame(kurtosis_rows),
        autocorrelation=pd.concat(autocorrelation, ignore_index=True),
    )


def plot_workspace_layers(
    geometry: LayerGeometry,
    readouts: dict[str, pd.DataFrame],
    *,
    n_layers: int,
    title: str,
) -> tuple[Figure, Figure]:
    """Return a Figure 27 CKA heatmap and a Figure 28 four-panel diagnostic."""
    import matplotlib.pyplot as plt

    def depth(x):
        return 100 * np.asarray(x) / max(n_layers - 1, 1)

    fig_cka, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    im = ax.imshow(geometry.cka, origin="lower", cmap="viridis", vmin=0, vmax=1)
    ticks = np.unique(
        np.linspace(0, len(geometry.layers) - 1, min(6, len(geometry.layers))).astype(
            int
        )
    )
    labels = [f"{depth(geometry.layers[i]):.0f}" for i in ticks]
    ax.set(
        xticks=ticks,
        xticklabels=labels,
        yticks=ticks,
        yticklabels=labels,
        xlabel="Layer (reindexed 0–100)",
        ylabel="Layer (reindexed 0–100)",
        title=f"Centered kernel alignment of J-lens\n{title}",
    )
    fig_cka.colorbar(im, ax=ax, label="CKA similarity")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    specifications = [
        (
            readouts["accuracy"],
            "k",
            "accuracy",
            "(a) Next-token prediction",
            "Top-k accuracy",
            "k",
        ),
        (
            readouts["kurtosis"],
            "percentile",
            "excess_kurtosis",
            "(b) J-lens readout kurtosis",
            "Excess kurtosis",
            "Percentile",
        ),
        (
            readouts["autocorrelation"],
            "lag",
            "delta_log_p",
            "(c) Top-1 autocorrelation",
            "Δ log p (vs shuffle null)",
            "Lag",
        ),
        (
            geometry.dimensions,
            "variance",
            "dimension_fraction",
            "(d) J-space dimensionality",
            "Fraction of residual dimensions",
            "Variance",
        ),
    ]
    for ax, (frame, group, value, heading, ylabel, legend) in zip(
        axes.flat, specifications, strict=True
    ):
        groups = list(frame.groupby(group, sort=True))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(groups)))
        for (label, rows), color in zip(groups, colors, strict=True):
            rows = rows.sort_values("layer")
            ax.plot(depth(rows.layer), rows[value], color=color, label=f"{label:g}")
        ax.set(
            title=heading,
            xlabel="Layer (reindexed 0–100)",
            ylabel=ylabel,
            xlim=(0, 100),
        )
        ax.grid(alpha=0.2)
        ax.legend(title=legend, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    axes[0, 0].set_ylim(0, 1.02)
    axes[1, 1].set_ylim(0, 1.02)
    fig.suptitle(title)
    return fig_cka, fig
