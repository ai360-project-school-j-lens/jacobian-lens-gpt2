"""J-lens on block outputs (residual deltas) vs on the residual stream.

Block l writes ``delta_l = h_l - h_{l-1}`` (``h_{-1}`` = token + position
embedding). It enters the stream where ``J_l = d h_final / d h_l`` is taken,
so its first-order effect on the final residual is ``J_l delta_l``.

Readouts (``mode``):
  full    -- ``lm_head(ln_f(x))``, as the notebooks do for the residual
  linear  -- ``W (gamma * center(x))``: the differential of the final norm
             (scale 1/sigma is the same for every token, so ranks ignore it;
             the ln_f bias, a fixed vocabulary prior, is dropped)
Variants: {logit, J} x {resid, delta} x {full, linear}. logit/delta/linear is
direct logit attribution of each block. The model-output row is shared.

Writes ``deltas_summary.csv`` (scores), ``deltas_contrasts.csv`` (paired item
bootstrap CIs) and ``deltas_layers.png`` under ``--out``.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convergence"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cilib import REPO_DIR, load_eval_cache, load_model

from jlens.evaluation import DATASETS, load_eval, readout_position
from jlens.lens import JacobianLens

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
parser.add_argument("--lens", type=Path, default=None, help="default: <run-dir>/snapshot_lens_134.pt")
parser.add_argument("--out", type=Path, default=None, help="default: <run-dir>/deltas")
parser.add_argument("--item-boot", type=int, default=2000)
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()
out = args.out or args.run_dir / "deltas"
out.mkdir(parents=True, exist_ok=True)
device = "cuda"
rng = np.random.default_rng(2)

model = load_model(device)
cache = load_eval_cache(args.run_dir / "eval_cache.pt")
H, words = cache["H"], cache["words"]
n_inner = H.shape[1] - 1
lens = JacobianLens.load(str(args.lens or args.run_dir / "snapshot_lens_134.pt"))


# ---- embedding residual (input of block 0) at every readout position ----
@torch.no_grad()
def embeddings():
    captured = {}

    def hook(module, hook_args, hook_kwargs):
        captured["x"] = hook_args[0] if hook_args else hook_kwargs["hidden_states"]

    def output_hook(module, hook_args, output):
        captured["h0"] = output[0] if isinstance(output, tuple) else output

    handles = [model.layers[0].register_forward_pre_hook(hook, with_kwargs=True),
               model.layers[0].register_forward_hook(output_hook)]
    rows, h0 = [], []
    try:
        for dataset in DATASETS:
            for item in load_eval(str(REPO_DIR), dataset):
                input_ids = model.encode(item["prompt"].rstrip())
                position = readout_position(model.tokenizer, input_ids[0].tolist(), dataset)
                model.forward(input_ids)
                rows.append(captured["x"][0, position].float().cpu())
                h0.append(captured["h0"][0, position].float().cpu())
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack(rows), torch.stack(h0)


E, h0 = embeddings()
# Same items in the same order as the cache: block-0 outputs must match the cached h_0.
torch.testing.assert_close(h0, H[:, 0], rtol=1e-4, atol=1e-4)
residual = H[:, :n_inner]  # h_0..h_{n_inner-1}
delta = residual - torch.cat([E[:, None], H[:, : n_inner - 1]], dim=1)
print("mean |delta| / |h| per layer (every 8th):",
      np.round((delta.norm(dim=-1) / residual.norm(dim=-1)).mean(0).numpy()[::8], 3), flush=True)

# ---- readouts and ranks ----
W = model._lm_head.weight.float()
gamma = model._final_norm.weight.float()
n_ids = words.ids.map(len).max()
ids = torch.zeros(len(words), n_ids, dtype=torch.long)
mask = torch.zeros(len(words), n_ids, dtype=torch.bool)
for row, word_ids in enumerate(words.ids):
    ids[row, : len(word_ids)] = torch.tensor(word_ids)
    mask[row, : len(word_ids)] = True
ids, mask = ids.to(device), mask.to(device)
word_item = torch.tensor(words.item_index.values, device=device)


@torch.no_grad()
def ranks_from_logits(logits):
    word_logits = logits[word_item]
    best = word_logits.gather(1, ids).masked_fill(~mask, -torch.inf).max(1).values
    return ((word_logits > best[:, None]).sum(1) + 1).cpu().numpy()


@torch.no_grad()
def readout(x, mode):
    if mode == "full":
        return model.unembed(x).float()
    centered = x - x.mean(-1, keepdim=True)
    return (centered * gamma) @ W.T


final_ranks = ranks_from_logits(model.unembed(H[:, -1].to(device)).float())
ranks = {}
for lens_name in ("logit", "J"):
    for source_name, source in (("resid", residual), ("delta", delta)):
        for mode in ("full", "linear"):
            R = np.zeros((len(words), n_inner + 1), dtype=np.int64)
            for layer in range(n_inner):
                x = source[:, layer].to(device)
                if lens_name == "J":
                    x = x @ lens.jacobians[layer].to(device).float().T
                R[:, layer] = ranks_from_logits(readout(x, mode))
            R[:, -1] = final_ranks
            ranks[f"{lens_name}/{source_name}/{mode}"] = R
            print("done", lens_name, source_name, mode, flush=True)
np.savez_compressed(out / "deltas_ranks.npz", **{k.replace("/", "__"): v for k, v in ranks.items()})

# ---- per-item scores ----
items_of = {}
for dataset in DATASETS:
    items_of[dataset] = np.unique(words.item_index.values[(words.dataset == dataset).values])


def per_item(R, dataset, kind, metric):
    """[n_items_of_dataset] per-item score; NaN where the item has no word of this kind."""
    sel = ((words.dataset == dataset) & (words.kind == kind)).values
    inner = R[sel, :-1].min(1)
    x = np.log10(inner) if metric == "log10 rank" else (inner <= int(metric.split("@")[1])).astype(float)
    item_pos = np.searchsorted(items_of[dataset], words.item_index.values[sel])
    total = np.bincount(item_pos, weights=x, minlength=len(items_of[dataset]))
    count = np.bincount(item_pos, minlength=len(items_of[dataset]))
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


metrics = [f"pass@{k}" for k in (1, args.k, 100)] + ["log10 rank"]
rows = []
for name, R in ranks.items():
    for dataset in DATASETS:
        for kind in ("intermediate", "control", "target"):
            for metric in metrics:
                score = per_item(R, dataset, kind, metric)
                if np.isnan(score).all():
                    continue
                rows.append({"variant": name, "dataset": dataset, "kind": kind,
                             "metric": f"inner {metric}", "value": np.nanmean(score)})
summary = pd.DataFrame(rows)
summary.to_csv(out / "deltas_summary.csv", index=False)

# ---- paired item-bootstrap contrasts ----
CONTRASTS = [
    ("J/delta/linear", "J/resid/full"),
    ("J/delta/full", "J/resid/full"),
    ("J/resid/linear", "J/resid/full"),
    ("J/delta/linear", "logit/delta/linear"),
    ("J/delta/linear", "logit/resid/full"),
    ("logit/delta/linear", "logit/resid/full"),
]
metric = f"pass@{args.k}"
contrast_rows = []
for dataset in DATASETS:
    n = len(items_of[dataset])
    draws = rng.integers(0, n, size=(args.item_boot, n))
    scores = {}
    for name, R in ranks.items():
        inter = per_item(R, dataset, "intermediate", metric)
        control = per_item(R, dataset, "control", metric)
        scores[name] = {"intermediate": inter, "control": control, "margin": inter - control}
    for a, b in CONTRASTS:
        for quantity in ("intermediate", "control", "margin"):
            diff = scores[a][quantity] - scores[b][quantity]
            if np.isnan(diff).all():
                continue
            bs = np.nanmean(diff[draws], axis=1)
            contrast_rows.append({
                "dataset": dataset, "contrast": f"{a} − {b}", "quantity": quantity,
                "estimate": np.nanmean(diff), "CI lo": np.nanpercentile(bs, 2.5), "CI hi": np.nanpercentile(bs, 97.5),
            })
contrasts = pd.DataFrame(contrast_rows)
contrasts.to_csv(out / "deltas_contrasts.csv", index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
for kind in ("intermediate", "control"):
    table = summary[(summary.kind == kind) & (summary.metric == f"inner {metric}")].pivot(
        index="variant", columns="dataset", values="value")[DATASETS]
    print(f"\n=== inner {metric}, {kind} ===")
    print(table.round(3).to_string())
table = summary[(summary.kind == "intermediate") & (summary.metric == "inner log10 rank")].pivot(
    index="variant", columns="dataset", values="value")[DATASETS]
print("\n=== mean log10 inner rank, intermediate ===")
print(table.round(3).to_string())
print(f"\n=== paired contrasts, inner {metric} (95% item-bootstrap CI) ===")
contrasts["text"] = [f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" for e, lo, hi in
                     zip(contrasts.estimate, contrasts["CI lo"], contrasts["CI hi"], strict=True)]
print(contrasts.pivot_table(index=["contrast", "quantity"], columns="dataset", values="text",
                            aggfunc="first")[DATASETS].to_string())

# ---- best-layer histogram and per-layer hit rate ----
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
styles = {"J/resid/full": ("#2a78d6", "-"), "J/delta/linear": ("#eb6834", "-"),
          "logit/delta/linear": ("#1baf7a", "-")}
for ax, dataset in zip(axes.flat, DATASETS, strict=False):
    for name, (color, _) in styles.items():
        for kind, ls in (("intermediate", "-"), ("control", "--")):
            sel = ((words.dataset == dataset) & (words.kind == kind)).values
            if not sel.any():
                continue
            hit = (ranks[name][sel, :-1] <= args.k).mean(0)
            ax.plot(range(n_inner), hit, color=color, ls=ls, lw=2 if kind == "intermediate" else 1.2,
                    label=f"{name}, {kind}")
    ax.set_title(dataset)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("layer")
for ax in axes[:, 0]:
    ax.set_ylabel(f"share of words with rank ≤ {args.k}")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out / "deltas_layers.png", dpi=130)
print("\nwritten", out)
