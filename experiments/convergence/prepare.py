"""Cache eval readout residuals and check the cached path against ``evaluate_paired``.

Writes ``eval_cache.pt`` and, for the snapshot lens, the reference
``words``/``items`` frames of both lenses from the notebook evaluator
(``baseline_words.pkl``, ``baseline_items.pkl``).
"""

import argparse
from pathlib import Path

import numpy as np
from cilib import (
    REPO_DIR,
    Ranker,
    build_eval_cache,
    load_model,
    save_eval_cache,
    transport_with,
)

from jlens.evaluation import DATASETS, evaluate_paired, load_eval, pass_at_k
from jlens.lens import JacobianLens

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
args = parser.parse_args()

model = load_model()
cache = build_eval_cache(model)
save_eval_cache(cache, args.run_dir / "eval_cache.pt")
print("eval cache:", tuple(cache["H"].shape), len(cache["words"]), "words", flush=True)

lens = JacobianLens.load(str(args.run_dir / "snapshot_lens_134.pt"))
ranker = Ranker(model, cache)
H = cache["H"]
cached = {
    "J-lens": ranker.words_frame(transport_with(lens.jacobians, H)),
    "logit lens": ranker.words_frame(H[:, :-1].transpose(0, 1)),
}

evals = {dataset: load_eval(str(REPO_DIR), dataset) for dataset in DATASETS}
words, items = evaluate_paired(model, lens, evals)
words.to_pickle(args.run_dir / "baseline_words.pkl")
items.to_pickle(args.run_dir / "baseline_items.pkl")

for name, frame in cached.items():
    reference = words[words.lens == name].reset_index(drop=True)
    assert (reference[["dataset", "item", "kind", "word", "role"]].values
            == frame[["dataset", "item", "kind", "word", "role"]].values).all()
    a, b = np.stack(reference.ranks), np.stack(frame.ranks)
    log_diff = np.abs(np.log10(a) - np.log10(b))
    print(f"{name}: exact rank match {np.mean(a == b):.4f}, max |log10 diff| {log_diff.max():.3f}, "
          f"best_rank match {np.mean(reference.best_rank.values == frame.best_rank.values):.4f}")
    ref_scores = pass_at_k(reference, ks=[10], layers=slice(None, -1)).set_index(["dataset", "kind"]).score
    new_scores = pass_at_k(frame, ks=[10], layers=slice(None, -1)).set_index(["dataset", "kind"]).score
    print("  max |inner pass@10 diff|:", float((ref_scores - new_scores).abs().max()))
