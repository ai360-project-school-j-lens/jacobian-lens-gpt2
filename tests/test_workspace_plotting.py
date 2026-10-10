"""Regression coverage for workspace figure export and SVG failure."""

import matplotlib
import pytest

matplotlib.use("Agg")

from jlens.workspace_plotting import save_workspace_figure  # noqa: E402


def test_exports_png_and_svg(tmp_path):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots()
    ax.set_title("Workspace export")
    ax.imshow([[1.0, 0.5], [0.5, 1.0]], vmin=0, vmax=1, cmap="viridis")
    try:
        paths = save_workspace_figure(figure, tmp_path / "cka")
        assert paths["png"].read_bytes().startswith(b"\x89PNG")
        assert "<svg" in paths["svg"].read_text()
    finally:
        plt.close(figure)


@pytest.mark.parametrize("missing", ["get_fontfeatures", "get_language"])
def test_mixed_svg_text_api_keeps_png_and_does_not_write_partial_svg(
    tmp_path,
    monkeypatch,
    missing,
):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_svg import RendererSVG

    figure, ax = plt.subplots()
    ax.set_title("Export survives SVG version mismatch")

    def broken_svg(*args, **kwargs):
        raise AttributeError(f"'Text' object has no attribute '{missing}'")

    monkeypatch.setattr(RendererSVG, "_draw_text_as_path", broken_svg)
    try:
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            with pytest.warns(RuntimeWarning, match="PNG saved; SVG skipped"):
                paths = save_workspace_figure(figure, tmp_path / "fallback")
        assert paths["png"].read_bytes().startswith(b"\x89PNG")
        assert paths["svg"] is None
        assert not (tmp_path / "fallback.svg").exists()
        figure.canvas.draw()  # The PNG/inline rendering canvas is still usable.
    finally:
        plt.close(figure)


def test_unrelated_export_errors_are_not_suppressed(tmp_path):
    from unittest.mock import Mock

    figure = Mock()
    figure.savefig.side_effect = [None, AttributeError("unrelated backend error")]
    with pytest.raises(AttributeError, match="unrelated"):
        save_workspace_figure(figure, tmp_path / "error")
