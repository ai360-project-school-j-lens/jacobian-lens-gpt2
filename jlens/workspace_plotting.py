"""Workspace figure export, independent of measurement caches."""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


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
