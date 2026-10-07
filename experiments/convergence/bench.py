"""Time one full per-prompt Jacobian at several dim_batch values (fp32 and fp16 autocast).

Also writes the fixed prompt list (snapshot prompts first, then the extension).
"""

import argparse
import time
from pathlib import Path

import torch
from cilib import build_prompt_list, load_model

from jlens.fitting import jacobian_for_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("/workspace/runs/ci"))
parser.add_argument("--dim-batches", type=int, nargs="+", default=[4, 8, 16])
args = parser.parse_args()

prompts = build_prompt_list(args.run_dir / "prompts.jsonl")
print(prompts.groupby(["part", "source"]).size().unstack(0), flush=True)

model = load_model()
lengths = [model.encode(text, max_length=128).shape[1] for text in prompts.text[:20]]
prompt = prompts.text[max(range(20), key=lengths.__getitem__)]
layers = list(range(model.n_layers - 1))

reference = None
for autocast in (False, True):
    for dim_batch in args.dim_batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                J, seq_len, n_valid = jacobian_for_prompt(model, prompt, layers, dim_batch=dim_batch, max_seq_len=128)
        except torch.OutOfMemoryError:
            print(f"autocast={autocast} dim_batch={dim_batch}: OOM", flush=True)
            continue
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        line = (f"autocast={autocast} dim_batch={dim_batch} seq_len={seq_len}: {seconds:.1f}s, "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        if reference is None:
            reference = J
        else:
            rel = [((J[l] - reference[l]).norm() / reference[l].norm()).item() for l in layers]
            line += f", max rel diff vs first fp32 {max(rel):.2e} (L{layers[rel.index(max(rel))]})"
        print(line, flush=True)
