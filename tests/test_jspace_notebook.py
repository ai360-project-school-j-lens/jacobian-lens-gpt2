"""Offline checks of the model-agnostic J-space notebook's per-row metric helpers."""

import json
from itertools import product
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from jlens.workspace import jspace_pursuit

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/model_agnostic/latent_bridge_v2_jspace_bootstrap.ipynb"
POSITIONS = ["subject", "country token", "last", "named"]


@pytest.fixture(scope="module")
def helpers():
    cells = {cell.id: cell.source for cell in nbformat.read(NOTEBOOK, as_version=4).cells}
    languages = json.loads((ROOT / "data/experiments/latent-bridge-v2-candidates.json").read_text())["languages"]
    state = dict(
        torch=torch, F=F, pd=pd, jspace_pursuit=jspace_pursuit, POSITIONS=POSITIONS,
        LATENT_POSITIONS=POSITIONS[:3], COUNTRIES=sorted(languages), LANG=languages,
        K_MAX=6, K_GRID=[2, 6], ROW_CHUNK=3, REFIT_STEPS=50, MAX_SEQ_LEN=128,
    )
    exec(cells["jspace-helpers"], state)
    return state


def test_depth_bands_reproduce_the_gpt2_xl_bands_and_exclude_the_final_block(helpers):
    bands = helpers["depth_bands"](48)
    assert list(bands) == ["8-16", "16-24", "24-32", "32-40", "40-47"]
    assert bands["40-47"] == list(range(40, 47))
    for n_layers in [6, 12, 28, 36, 62]:
        layers = [layer for band in helpers["depth_bands"](n_layers).values() for layer in band]
        assert layers == sorted(set(layers)) and max(layers) == n_layers - 2


def _sites(helpers, rows):
    countries = helpers["COUNTRIES"]
    frame = pd.DataFrame(rows, columns=["family", "country", "position"])
    frame["control"] = [countries[(countries.index(c) + 11) % len(countries)] for c in frame.country]
    return frame, helpers["row_groups"](frame)


def test_row_consistency_matches_a_pairwise_loop(helpers):
    families = ["city", "brand", "food"]
    countries = ["Spain", "Mexico", "Peru", "Japan", "Brazil", "Portugal"]
    rows = [(f, c, "subject") for f, c in product(families, countries)] + [("city", "Spain", "subject")]
    sites, groups = _sites(helpers, rows)
    X = torch.randn(len(sites), 8, generator=torch.Generator().manual_seed(0))
    mask = torch.ones(len(sites), dtype=torch.bool)
    overall, by_language = helpers["row_consistency"](X, groups, mask)
    lang = helpers["LANG"]
    cos = F.normalize(X, dim=1) @ F.normalize(X, dim=1).T
    for i, (fi, ci, _) in enumerate(rows):
        same = [cos[i, j] for j, (fj, cj, _) in enumerate(rows) if fj != fi and cj == ci]
        diff = [cos[i, j] for j, (fj, cj, _) in enumerate(rows) if fj != fi and cj != ci]
        assert overall[i] == pytest.approx(float(np.mean(same) - np.mean(diff)), abs=1e-5)
        same_language = [cos[i, j] for j, (fj, cj, _) in enumerate(rows)
                         if fj != fi and cj != ci and lang[cj] == lang[ci]]
        if ci == "Japan":
            assert torch.isnan(by_language[i])
        else:
            assert by_language[i] == pytest.approx(float(np.mean(same) - np.mean(same_language)), abs=1e-5)


def test_named_hits_use_centroids_language_groups_and_report_missing_countries(helpers):
    rows = [("named", c, "named") for c in ["Spain", "Spain", "Mexico", "Japan"]]
    rows += [("city", c, "subject") for c in ["Spain", "Mexico", "Japan", "Peru"]]
    _, groups = _sites(helpers, rows)
    spain, mexico, japan = torch.eye(4)[:3]
    X = torch.stack([spain, spain, mexico, japan,
                     spain + 0.1 * mexico, spain + 0.2 * japan, japan, mexico])
    hit, within = helpers["named_hits"](X, groups)
    assert torch.isnan(hit[:4]).all() and torch.isnan(within[:4]).all()  # named rows
    assert hit[4:7].tolist() == [1.0, 0.0, 1.0]
    # Mexico's row is nearest to Spain, overall and within the Spanish-speaking countries.
    assert within[5] == 0.0 and within[4] == 1.0 and torch.isnan(within[6])  # Japan: no shared language
    assert torch.isnan(hit[7]) and torch.isnan(within[7])  # Peru has no named centroid


def test_occupancy_dispersion_and_chunked_decomposition(helpers):
    explained = torch.tensor([[0.5, 0.6, 0.605, 0.7], [0.1, 0.2, 0.3, 0.4]])
    assert helpers["occupancy"](explained, 0.01).tolist() == [2, 4]
    Y = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    dispersion = helpers["relative_dispersion"](Y, torch.tensor([0, 0, 1, 1]), 2)
    torch.testing.assert_close(dispersion, torch.tensor([0.0, 1.0]))
    generator = torch.Generator().manual_seed(3)
    D = F.normalize(torch.randn(20, 8, generator=generator), dim=1)
    X = torch.randn(7, 8, generator=generator)
    explained, support, snapshots = helpers["decompose"](X, D)
    whole = jspace_pursuit(X, D, 6, refit_steps=50, snapshots=[2, 6])
    torch.testing.assert_close(explained, whole.explained)
    assert torch.equal(support, whole.support)
    for k in [2, 6]:
        torch.testing.assert_close(snapshots[k][0], whole.snapshots[k][0])
