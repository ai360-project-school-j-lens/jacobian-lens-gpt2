"""Workspace CKA controls and figure export, independent of measurement caches."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from jlens.workspace_layers import LayerGeometry

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def logit_lens_geometry(geometry: LayerGeometry, *, final_layer: int) -> LayerGeometry:
    """Derive the exact fixed-unembedding control from the final J-lens row.

    ``workspace_geometry`` uses identity transport at the final block. Thus its
    final geometry is exactly that of the logit lens, whose vectors are ``W_U``
    at every block. Nondegenerate linear CKA is one for every pair; the PCA
    dimension fractions equal the final row at every layer. No fitting, forward
    passes, vocabulary allocations, or eigendecompositions are needed. This
    measures vector geometry, not CKA between context-dependent activations.
    """
    if not geometry.layers or geometry.layers[-1] != final_layer:
        raise ValueError("Geometry must include the model's final block")
    n = len(geometry.layers)
    if geometry.cka.shape != (n, n):
        raise ValueError("CKA shape must match selected layers")
    final_self = geometry.cka[-1, -1]
    if not np.isnan(final_self) and not np.isclose(final_self, 1, atol=2e-5):
        raise ValueError("Final CKA must be unit self-similarity or undefined")
    final_dimensions = geometry.dimensions.query("layer == @final_layer")
    if final_dimensions.empty:
        raise ValueError("Geometry lacks final-layer dimension fractions")
    dimensions = pd.concat(
        [final_dimensions.assign(layer=layer) for layer in geometry.layers],
        ignore_index=True,
    )
    # A constant/zero unembedding has undefined CKA, not perfect alignment.
    cka = np.full((n, n), np.nan if np.isnan(final_self) else 1.0)
    return LayerGeometry(list(geometry.layers), cka, dimensions)


def plot_cka_comparison(
    geometry: LayerGeometry, *, n_layers: int, title: str
) -> Figure:
    """Compare J-lens and fixed-unembedding logit-lens CKA on one color scale."""
    import matplotlib.pyplot as plt

    control = logit_lens_geometry(geometry, final_layer=n_layers - 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    ticks = np.unique(
        np.linspace(0, len(geometry.layers) - 1, min(6, len(geometry.layers))).astype(
            int
        )
    )
    labels = [f"{100 * geometry.layers[i] / max(n_layers - 1, 1):.0f}" for i in ticks]
    for ax, result, name in zip(
        axes,
        (geometry, control),
        ("J-lens", "Logit lens (fixed unembedding)"),
        strict=True,
    ):
        image = ax.imshow(result.cka, origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set(
            xticks=ticks,
            xticklabels=labels,
            yticks=ticks,
            yticklabels=labels,
            xlabel="Layer (reindexed 0–100)",
            ylabel="Layer (reindexed 0–100)",
            title=name,
        )
    figure.colorbar(image, ax=list(axes), label="CKA similarity", shrink=0.8)
    figure.suptitle(f"Centered kernel alignment of lens vectors\n{title}")
    return figure


def save_workspace_figure(
    figure: Figure, stem: str | Path, *, dpi: int = 180
) -> dict[str, Path | None]:
    """Save PNG and SVG; retain PNG if SVG encounters the mixed-Text API error.

    Render each format in memory before writing it, so a backend exception does
    not leave a partial file. Only missing Text font-feature/language methods
    are handled; unrelated failures still raise. A returned ``svg=None`` means
    no SVG was exported in this call, even if an older export already exists.
    Restarting the runtime after installing Matplotlib resolves mixed imports.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path | None] = {}
    for extension in ("png", "svg"):
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format=extension, dpi=dpi, bbox_inches="tight")
        except AttributeError as error:
            if extension != "svg" or not any(
                f"has no attribute '{attribute}'" in str(error)
                for attribute in ("get_fontfeatures", "get_language")
            ):
                raise
            warnings.warn(
                "PNG saved; SVG skipped because Matplotlib's loaded Text class "
                "and SVG backend have incompatible APIs. Restart the runtime "
                "after installation, then rerun using cached measurements. "
                f"Original SVG error: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            exported[extension] = None
            continue
        destination = stem.parent / f"{stem.name}.{extension}"
        destination.write_bytes(buffer.getvalue())
        exported[extension] = destination
    return exported
