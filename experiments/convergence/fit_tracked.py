"""Fit the J-lens prompt by prompt, keeping what every prompt contributes to the eval readout.

For snapshot prompts (the 134 behind the saved lens) each prompt's
``T_p = H @ J_p.T`` [n_inner, n_items, d_model] is saved in fp16 as
``T/p{idx:04}.pt``. Extension prompts are summed in groups of ``--group``
(``T/g{first:04}-{last:04}.pt``) to save disk. Every file records the prompt
count it sums, so any subset or order of files gives a lens readout.

A running sum of the full Jacobians is checkpointed for resume, and the lens
is saved at ``--milestones`` prompt counts. Rerunning resumes after the last
complete file.
"""

import argparse
import logging
import math
import time
from pathlib import Path

import torch
from cilib import build_prompt_list, load_eval_cache, load_model

from jlens.fitting import jacobian_for_prompt
from jlens.lens import JacobianLens

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
parser.add_argument("--dim-batch", type=int, default=8)
parser.add_argument("--max-seq-len", type=int, default=128)
parser.add_argument("--group", type=int, default=16, help="extension prompts per saved T file")
parser.add_argument("--stop", type=int, default=None, help="stop after this many prompts")
parser.add_argument("--milestones", type=int, nargs="*", default=[134, 268, 530])
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fit_tracked")
logging.getLogger("jlens.fitting").setLevel(logging.WARNING)

run_dir = args.run_dir
t_dir = run_dir / "T"
t_dir.mkdir(parents=True, exist_ok=True)
prompts = build_prompt_list(run_dir / "prompts.jsonl")
n_snapshot = int((prompts.part == "snapshot").sum())
stop = len(prompts) if args.stop is None else min(args.stop, len(prompts))

# Files: one per snapshot prompt, then groups of extension prompts.
units = [[idx] for idx in range(n_snapshot)]
units += [list(range(start, min(start + args.group, len(prompts)))) for start in range(n_snapshot, len(prompts), args.group)]


def unit_path(unit):
    return t_dir / (f"p{unit[0]:04}.pt" if len(unit) == 1 and unit[0] < n_snapshot else f"g{unit[0]:04}-{unit[-1]:04}.pt")


model = load_model()
layers = list(range(model.n_layers - 1))
H = load_eval_cache(run_dir / "eval_cache.pt")["H"].cuda()  # [n_items, n_layers, d]

state_path = run_dir / "jsum.pt"
if state_path.exists():
    state = torch.load(state_path, weights_only=True)
    jsum, n_fit, done_idx = state["jsum"], state["n_fit"], state["done_idx"]
else:
    jsum = {layer: torch.zeros(model.d_model, model.d_model) for layer in layers}
    n_fit, done_idx = 0, 0
log.info("resume: %d prompts fitted, next index %d", n_fit, done_idx)

for unit in units:
    if unit[-1] < done_idx:
        if not unit_path(unit).exists():
            raise RuntimeError(f"{unit_path(unit)} is missing although the J checkpoint covers it")
        continue
    if unit[0] >= stop:
        break
    T_sum = torch.zeros(len(layers), H.shape[0], model.d_model, device="cuda")
    meta = []
    unit_jsum = {layer: torch.zeros_like(jsum[layer]) for layer in layers}
    for idx in unit:
        started = time.perf_counter()
        try:
            J, seq_len, n_valid = jacobian_for_prompt(
                model, prompts.text[idx], layers, dim_batch=args.dim_batch, max_seq_len=args.max_seq_len,
            )
        except ValueError as error:
            log.warning("skip prompt %d: %s", idx, error)
            continue
        norms = []
        for layer in layers:
            J_gpu = J[layer].cuda()
            T_sum[layer] += H[:, layer] @ J_gpu.T
            unit_jsum[layer] += J[layer]
            norms.append(J[layer].norm().item() / math.sqrt(model.d_model))
        meta.append({"idx": idx, "source": prompts.source[idx], "seq_len": seq_len, "n_valid": n_valid,
                     "jnorm": norms, "seconds": time.perf_counter() - started})
        log.info("prompt %d/%d (%s) seq_len=%d %.0fs", idx + 1, len(prompts), prompts.source[idx],
                 seq_len, meta[-1]["seconds"])
        del J
    tmp = unit_path(unit).with_suffix(".tmp")
    torch.save({"T_sum": T_sum.half().cpu(), "n": len(meta), "meta": meta}, tmp)
    tmp.replace(unit_path(unit))
    for layer in layers:
        jsum[layer] += unit_jsum[layer]
    previous, n_fit, done_idx = n_fit, n_fit + len(meta), unit[-1] + 1
    tmp = state_path.with_suffix(".tmp")
    torch.save({"jsum": jsum, "n_fit": n_fit, "done_idx": done_idx}, tmp)
    tmp.replace(state_path)
    for milestone in args.milestones:
        if previous < milestone <= n_fit:
            JacobianLens({layer: jsum[layer] / n_fit for layer in layers}, n_prompts=n_fit,
                         d_model=model.d_model).save(str(run_dir / f"lens_n{n_fit}.pt"))
            log.info("saved lens at %d prompts", n_fit)
log.info("done: %d prompts", n_fit)
