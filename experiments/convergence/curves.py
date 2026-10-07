"""Learning curves and fit-corpus variability of the J-lens from the saved per-prompt ``T`` files.

Every "lens" here is a weight vector over the saved units (a unit = one snapshot
prompt or one group of extension prompts): readout = sum_u w_u T_u / sum_u w_u n_u.
All lenses are evaluated together, layer by layer, so each T slice is moved to
the GPU once.

Experiments (column ``experiment``):
  order     -- cumulative mean in the original fit order, N = 1..all
  perm      -- cumulative mean in random orders of the snapshot prompts (band)
  boot      -- source-stratified bootstrap of the snapshot prompts (N = 134)
  half      -- random disjoint halves of the snapshot (67 + 67), both halves kept
  ext       -- snapshot + extension groups, cumulative
  ext_only  -- extension prompts only (an independent corpus), cumulative
  snapshot  -- the saved 134-prompt lens itself, for the reproduction check
  logit     -- logit lens
Outputs ``curves.parquet`` (long scores) and ``ranks_<experiment>.npy``.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from cilib import MODEL_ID, load_eval_cache

from jlens.lens import JacobianLens

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
parser.add_argument("--perms", type=int, default=30)
parser.add_argument("--boot", type=int, default=200)
parser.add_argument("--halves", type=int, default=50)
parser.add_argument("--device", default="cuda")
parser.add_argument("--chunk", type=int, default=4, help="lenses per unembed batch")
parser.add_argument("--out-suffix", default="")
parser.add_argument("--snapshot-only", action="store_true", help="ignore extension groups")
args = parser.parse_args()
device = args.device
rng = np.random.default_rng(0)


class Unembed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hf = transformers.GPT2LMHeadModel.from_pretrained(MODEL_ID)
        self.ln_f, self.lm_head = hf.transformer.ln_f, hf.lm_head

    def forward(self, x):
        return self.lm_head(self.ln_f(x))


unembed = Unembed().to(device).eval()
cache = load_eval_cache(args.run_dir / "eval_cache.pt")
H, words = cache["H"], cache["words"]
n_inner = H.shape[1] - 1

# ---- units ----
# A unit file is complete once the J checkpoint covers it (the fit writes T first).
done_idx = torch.load(args.run_dir / "jsum.pt", weights_only=True, map_location="cpu")["done_idx"]
files = [f for f in sorted((args.run_dir / "T").glob("*.pt"))
         if int(f.stem[1:].split("-")[-1]) < done_idx and not (args.snapshot_only and f.name.startswith("g"))]
units = [torch.load(f, weights_only=False) for f in files]
is_snap = np.array([f.name.startswith("p") for f in files])
n_unit = np.array([u["n"] for u in units], dtype=float)
source = np.array([u["meta"][0]["source"] if u["n"] else "" for u in units])
T = torch.stack([u["T_sum"] for u in units])  # [U, n_inner, n_items, d] fp16
del units
snap = np.flatnonzero(is_snap)
ext = np.flatnonzero(~is_snap)
print(f"{len(snap)} snapshot units, {len(ext)} extension units ({int(n_unit[ext].sum())} prompts)", flush=True)

# ---- lenses: (experiment, N, rep, weights) ----
lenses = []


def add(experiment, rep, chosen):
    w = np.zeros(len(files))
    np.add.at(w, chosen, 1.0)
    lenses.append({"experiment": experiment, "N": int((w * n_unit).sum()), "rep": rep, "w": w})


for n in range(1, len(snap) + 1):
    add("order", 0, snap[:n])
grid = [4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 134]
for r in range(args.perms):
    order = rng.permutation(snap)
    for n in grid:
        if n <= len(snap):
            add("perm", r, order[:n])
by_source = {s: snap[source[snap] == s] for s in np.unique(source[snap])}
for r in range(args.boot):
    add("boot", r, np.concatenate([rng.choice(idx, len(idx)) for idx in by_source.values()]))
for r in range(args.halves):
    order = rng.permutation(snap)
    add("half", 2 * r, order[: len(snap) // 2])
    add("half", 2 * r + 1, order[len(snap) // 2:])
for g in range(1, len(ext) + 1):
    add("ext", 0, np.concatenate([snap, ext[:g]]))
    add("ext_only", 0, ext[:g])
W = torch.tensor(np.stack([lens["w"] for lens in lenses]), dtype=torch.float32)
W = W / (W * torch.tensor(n_unit, dtype=torch.float32)).sum(1, keepdim=True)  # weights -> mean

snapshot = JacobianLens.load(str(args.run_dir / "snapshot_lens_134.pt"))

# ---- word index tensors ----
n_ids = words.ids.map(len).max()
ids = torch.zeros(len(words), n_ids, dtype=torch.long)
mask = torch.zeros(len(words), n_ids, dtype=torch.bool)
for row, word_ids in enumerate(words.ids):
    ids[row, : len(word_ids)] = torch.tensor(word_ids)
    mask[row, : len(word_ids)] = True
ids, mask = ids.to(device), mask.to(device)
word_item = torch.tensor(words.item_index.values, device=device)


@torch.no_grad()
def ranks_of(residuals, row_chunk=512):
    """residuals [B, n_items, d] -> ranks [B, n_words], one unembed for the whole batch."""
    B, n_items = residuals.shape[:2]
    logits = unembed(residuals.to(device).float().reshape(B * n_items, -1))  # [B * n_items, vocab]
    rows = (torch.arange(B, device=device)[:, None] * n_items + word_item[None]).reshape(-1)
    all_ids, all_mask = ids.repeat(B, 1), mask.repeat(B, 1)
    ranks = torch.empty(len(rows), dtype=torch.int32, device=device)
    for start in range(0, len(rows), row_chunk):
        chunk = slice(start, start + row_chunk)
        word_logits = logits[rows[chunk]]
        best = word_logits.gather(1, all_ids[chunk]).masked_fill(~all_mask[chunk], -torch.inf).max(1).values
        ranks[chunk] = (word_logits > best[:, None]).sum(1).int() + 1
    return ranks.reshape(B, -1).cpu().numpy()


n_lens = len(lenses)
R = np.zeros((n_lens, len(words), n_inner + 1), dtype=np.int32)
R_snap = np.zeros((len(words), n_inner + 1), dtype=np.int32)
R_logit = np.zeros((len(words), n_inner + 1), dtype=np.int32)
for layer in range(n_inner):
    started = time.perf_counter()
    T_layer = T[:, layer].to(device).float()  # [U, n_items, d]
    for start in range(0, n_lens, args.chunk):
        w = W[start:start + args.chunk].to(device)
        R[start:start + args.chunk, :, layer] = ranks_of(torch.einsum("bu,uid->bid", w, T_layer))
    del T_layer
    h = H[:, layer].to(device)
    R_snap[:, layer] = ranks_of((h @ snapshot.jacobians[layer].to(device).float().T)[None])[0]
    R_logit[:, layer] = ranks_of(h[None])[0]
    print(f"layer {layer} done in {time.perf_counter() - started:.0f}s", flush=True)
final = ranks_of(H[:, -1].to(device)[None])[0]
R[:, :, -1] = final
R_snap[:, -1] = final
R_logit[:, -1] = final

# ---- scores ----
keys = words[["dataset", "kind"]].drop_duplicates().values.tolist()


def scores(ranks):
    """ranks [B, n_words, n_layers] -> rows of (dataset, kind, metric, value[B])."""
    inner = ranks[:, :, :-1].min(2)
    rows = []
    for dataset, kind in keys:
        sel = ((words.dataset == dataset) & (words.kind == kind)).values
        items = words.item_index.values[sel]
        _, inv = np.unique(items, return_inverse=True)
        n_per = np.bincount(inv)
        for k in (1, 10, 100):
            hit = (inner[:, sel] <= k).astype(float)
            per_item = np.stack([np.bincount(inv, weights=h) for h in hit]) / n_per
            rows.append((dataset, kind, f"inner pass@{k}", per_item.mean(1)))
        log_rank = np.log10(inner[:, sel])
        per_item = np.stack([np.bincount(inv, weights=h) for h in log_rank]) / n_per
        rows.append((dataset, kind, "mean log10 inner rank", per_item.mean(1)))
    return rows


meta = pd.DataFrame([{k: v for k, v in lens.items() if k != "w"} for lens in lenses])
frames = []
for dataset, kind, metric, values in scores(R):
    frames.append(meta.assign(dataset=dataset, kind=kind, metric=metric, value=values))
for name, ranks in (("snapshot", R_snap), ("logit", R_logit)):
    for dataset, kind, metric, values in scores(ranks[None]):
        frames.append(pd.DataFrame({"experiment": [name], "N": [134 if name == "snapshot" else 0], "rep": [0],
                                    "dataset": dataset, "kind": kind, "metric": metric, "value": values}))
curves = pd.concat(frames, ignore_index=True)
curves.to_parquet(args.run_dir / f"curves{args.out_suffix}.parquet")
np.save(args.run_dir / f"ranks_lenses{args.out_suffix}.npy", R)
np.save(args.run_dir / f"ranks_snapshot{args.out_suffix}.npy", R_snap)
np.save(args.run_dir / f"ranks_logit{args.out_suffix}.npy", R_logit)
meta.to_parquet(args.run_dir / f"lenses_meta{args.out_suffix}.parquet")

# Reproduction: refit of the 134 snapshot prompts vs the saved snapshot lens.
refit = R[meta.index[(meta.experiment == "order") & (meta.N == len(snap))][0]]
print("refit vs snapshot: exact rank match", np.mean(refit[:, :-1] == R_snap[:, :-1]),
      "inner_best match", np.mean(refit[:, :-1].min(1) == R_snap[:, :-1].min(1)))
print("saved", args.run_dir / f"curves{args.out_suffix}.parquet")
