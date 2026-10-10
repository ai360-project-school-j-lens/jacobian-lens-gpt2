"""Offline logit-lens controls and regression coverage for SVG export failure."""

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from jlens.evaluation import identity_lens  # noqa: E402
from jlens.lens import JacobianLens  # noqa: E402
from jlens.workspace_layers import workspace_geometry  # noqa: E402
from jlens.workspace_plotting import (  # noqa: E402
    logit_lens_geometry,
    plot_cka_comparison,
    save_workspace_figure,
)
from tests.tiny import TinyDecoder  # noqa: E402


def test_logit_control_matches_explicit_identity_geometry():
    model = TinyDecoder(n_layers=3)
    lens = JacobianLens(
        {0: torch.randn(8, 8), 1: torch.randn(8, 8)},
        n_prompts=1,
        d_model=8,
    )
    geometry = workspace_geometry(
        model,
        lens,
        model.lm_head.weight,
        layers=[0, 1],
        progress=False,
    )
    identity = workspace_geometry(
        model,
        identity_lens(model),
        model.lm_head.weight,
        layers=[0, 1],
        progress=False,
    )
    control = logit_lens_geometry(geometry, final_layer=2)
    np.testing.assert_allclose(control.cka, identity.cka, atol=2e-6)
    np.testing.assert_allclose(control.dimensions, identity.dimensions, atol=2e-6)
    assert np.array_equal(control.cka, np.ones((3, 3)))
    assert geometry.cka[0, -1] < 0.99  # J-lens retains its distinct geometry.


def test_zero_unembedding_control_remains_undefined():
    model = TinyDecoder()
    geometry = workspace_geometry(
        model,
        identity_lens(model),
        torch.ones(32, 8),
        layers=[0],
        progress=False,
    )
    control = logit_lens_geometry(geometry, final_layer=3)
    assert np.isnan(control.cka).all()
    assert control.dimensions.dimension_fraction.isna().all()
    with pytest.raises(ValueError, match="final block"):
        logit_lens_geometry(geometry, final_layer=2)


def test_comparison_exports_png_and_svg(tmp_path):
    import matplotlib.pyplot as plt

    model = TinyDecoder()
    geometry = workspace_geometry(
        model,
        identity_lens(model),
        model.lm_head.weight,
        layers=[0],
        progress=False,
    )
    figure = plot_cka_comparison(geometry, n_layers=4, title="CPU regression")
    try:
        assert len(figure.axes) == 3  # two heatmaps and one shared colorbar
        np.testing.assert_array_equal(
            figure.axes[1].images[0].get_array(), np.ones((2, 2))
        )
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
