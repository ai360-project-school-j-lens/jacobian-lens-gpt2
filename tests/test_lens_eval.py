"""Offline GPT-2 checks for the shared-activation dataset comparison."""

import contextlib
import io
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gpt2 import GPT2LensModel
from gpt2.lens_eval import (
    DATASETS,
    evaluate_all,
    evaluate_paired,
    identity_lens,
    jacobian_lens,
    load_eval,
    logit_lens,
)
from jlens.lens import JacobianLens

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks/jacobian_lens/jacobian_logit_lens_dataset.ipynb"


class _AsciiTokenizer:
    bos_token_id = 128

    def encode(self, text):
        return [ord(char) % 128 for char in text]

    def __call__(self, text):
        return SimpleNamespace(input_ids=self.encode(text))

    def decode(self, ids, **kwargs):
        return "".join(chr(token) if token < 128 else "<BOS>" for token in ids)


class TestPairedEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        plt.switch_backend("Agg")

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(13)
        config = GPT2Config(
            vocab_size=129, n_positions=128, n_embd=8, n_layer=3, n_head=2,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        )
        self.gpt = GPT2LensModel(GPT2LMHeadModel(config), _AsciiTokenizer())
        self.lens = JacobianLens(
            {layer: torch.eye(8) + 0.3 * torch.randn(8, 8) for layer in range(2)},
            n_prompts=2, d_model=8,
        )
        self.evals = {dataset: load_eval(str(REPO), dataset)[:2] for dataset in DATASETS}

    def assert_frames_match(self, paired, expected, name):
        for got, want in zip(paired, expected):
            got = got[got.lens == name].drop(columns="lens").reset_index(drop=True)
            # Pandas can infer optional strings differently when concatenating
            # per-dataset frames versus building the paired frame in one go.
            if "target" in got:
                got["target"] = got["target"].astype("string")
                want = want.assign(target=want.target.astype("string"))
            pd.testing.assert_frame_equal(got, want)

    def test_one_forward_per_item_and_matches_separate_lenses(self):
        # Transformers may install its own persistent hidden-state hooks.
        self.gpt.forward(self.gpt.encode("warmup"), output_hidden_states=True)
        initial_hooks = [dict(layer._forward_hooks) for layer in self.gpt.layers]
        with (
            patch.object(self.gpt, "forward", wraps=self.gpt.forward) as forward,
            patch.object(self.gpt, "encode", wraps=self.gpt.encode) as encode,
        ):
            paired = evaluate_paired(self.gpt, self.lens, self.evals)
        self.assertEqual(forward.call_count, sum(map(len, self.evals.values())))
        self.assertEqual(encode.call_count, forward.call_count)
        for layer, hooks in zip(self.gpt.layers, initial_hooks):
            self.assertEqual(dict(layer._forward_hooks), hooks)
        for name, lens_fn in (
            ("logit lens", partial(logit_lens, self.gpt)),
            ("J-lens", partial(jacobian_lens, self.gpt, self.lens)),
        ):
            expected = evaluate_all(self.gpt, lens_fn, self.evals)
            self.assert_frames_match(paired, expected, name)
        words, items = paired
        self.assertEqual(set(words.kind), {"intermediate", "control", "target"})
        self.assertEqual(set(items.lens), {"logit lens", "J-lens"})
        self.assertEqual(items[items.dataset == "poetry"].readout_token.unique().tolist(), ["\n"])

    def test_identity_and_shared_final_output(self):
        for lens in (identity_lens(self.gpt), self.lens):
            words, items = evaluate_paired(self.gpt, lens, self.evals)
            for frame in (words, items):
                left = frame[frame.lens == "logit lens"].drop(columns="lens").reset_index(drop=True)
                right = frame[frame.lens == "J-lens"].drop(columns="lens").reset_index(drop=True)
                if lens.n_prompts == 0:
                    pd.testing.assert_frame_equal(left, right)
            a, b = (words[words.lens == name] for name in ("logit lens", "J-lens"))
            np.testing.assert_array_equal(np.stack(a.ranks)[:, -1], np.stack(b.ranks)[:, -1])
            np.testing.assert_allclose(np.stack(items.agreement)[:, -1], 1)
            np.testing.assert_allclose(np.stack(items.kl_to_final)[:, -1], 0, atol=1e-6)

    def test_rejects_incomplete_or_wrong_width_lens_before_forward(self):
        for lens, message in (
            (JacobianLens({0: torch.eye(8)}, n_prompts=1, d_model=8), "missing inner layers"),
            (JacobianLens({0: torch.eye(4)}, n_prompts=1, d_model=4), "d_model"),
        ):
            with patch.object(self.gpt, "forward", wraps=self.gpt.forward) as forward:
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_paired(self.gpt, lens, self.evals)
                forward.assert_not_called()

    def test_notebook_analysis_cells_run_on_offline_gpt2(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        namespace = {
            "gpt": self.gpt, "fitted_lens": self.lens, "evals": self.evals,
            "LENS_NAMES": ("logit lens", "J-lens"), "LAYER_STRIDE": 1,
            "K": 5, "KS": [1, 5, 10, 100], "LAST": self.gpt.n_layers - 1,
        }
        run_analysis = False
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            patch.object(plt, "show"),
        ):
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if source.startswith("sanity_evals ="):
                    run_analysis = True
                # Import cell plus every cell from the identity check to conclusions.
                if source.startswith("import matplotlib") or run_analysis:
                    exec(compile(source, f"{NOTEBOOK.name}:cell {index}", "exec"), namespace)
                    plt.close("all")
        self.assertEqual(len(namespace["model_items"]), sum(map(len, self.evals.values())))
        self.assertEqual(len(namespace["head_to_head"]), len(namespace["inter"]) // 2)
        self.assertTrue(any("single-token" in line for line in namespace["lines"]))
        self.assertTrue(any("inner-layer pass@" in line for line in namespace["lines"]))
