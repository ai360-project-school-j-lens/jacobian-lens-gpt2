"""The workspace notebook loads prefitted lenses without entering fitting."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat
import pytest
import torch

import jlens
from jlens.evaluation import identity_lens
from tests.tiny import TinyDecoder

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks/model_agnostic/workspace_layers.ipynb"
)


@pytest.mark.parametrize("source", ["hub", "local"])
@pytest.mark.parametrize("failure", [None, "download", "width", "coverage", "nonfinite"])
def test_prefitted_notebook_never_fits(tmp_path, source, failure):
    if source == "local" and failure == "download":
        pytest.skip("Local loading makes no download request")
    model = TinyDecoder(n_layers=3)
    lens = identity_lens(model)
    if failure == "width":
        lens = identity_lens(TinyDecoder(n_layers=3, d_model=4))
    elif failure == "coverage":
        lens.jacobians.pop(0)
    elif failure == "nonfinite":
        lens.jacobians[0][0, 0] = torch.nan
    path = tmp_path / "matching-final.pt"
    lens.save(str(path))
    receipt = dict(resolved_revision="a" * 40, fitting_provenance="external")
    download = Mock(return_value=SimpleNamespace(path=str(path), to_dict=lambda: receipt))
    if failure == "download":
        download.side_effect = OSError("Network unavailable")
    forbidden = Mock(side_effect=AssertionError("Prefitted loading must not enter fitting"))
    namespace = dict(
        Path=Path, torch=torch, jlens=jlens, model=model,
        RUN_MODE="hf", LENS_SOURCE=source,
        EXISTING_LENS_PATH=path,
        HUB_LENS=dict(repo_id="offline/lenses", filename="matching-final.pt", revision="main"),
        download_lens_artifact=download, display=Mock(),
        DatasetFitRun=forbidden, fit_with_progress=forbidden,
        fit_prompt_loader=forbidden, benchmark_dim_batches=forbidden,
        fitted_lens="stale lens", readouts="stale readouts", geometry="stale geometry",
    )
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(c for c in notebook.cells if c.id == "workspace-load-lens")
    if failure:
        with pytest.raises(OSError if failure == "download" else ValueError):
            exec(cell.source, namespace)
        assert namespace["fitted_lens"] is None
    else:
        exec(cell.source, namespace)
        assert namespace["fitted_lens"].d_model == model.d_model
        assert namespace["lens_path"] == path
        assert namespace["lens_provenance"]["source"] == source
        if source == "hub":
            assert namespace["lens_provenance"]["resolved_revision"] == "a" * 40
    assert namespace["readouts"] is None
    assert namespace["geometry"] is None
    assert namespace["fit_run"] is None
    forbidden.assert_not_called()
    if source == "local":
        download.assert_not_called()
    else:
        download.assert_called_once_with(**namespace["HUB_LENS"])
