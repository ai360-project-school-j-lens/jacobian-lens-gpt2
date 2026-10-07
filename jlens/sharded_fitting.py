"""Fit disjoint prompt shards in separate processes and merge their lenses.

Workers load their own model and own their activation hooks/autograd graph.
Repeat a CUDA device to try concurrent workers on one GPU, or list distinct
devices to distribute the shards across GPUs. Account for all model copies,
graphs, and any model still resident in the calling notebook. Parallelism is
a throughput experiment, not a guarantee of speedup on a single GPU.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from jlens.fitting import fit, jacobian_for_prompt
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


def benchmark_dim_batches(
    model: LensModel,
    prompt: str,
    candidates: Sequence[int] = (8, 16, 32),
    *,
    max_seq_len: int = 128,
) -> list[dict]:
    """Time a complete per-prompt Jacobian at each batch size before a fit.

    CUDA peaks include the caller's resident model. Choose a representative
    long prompt; this is a local measurement, not a guaranteed corpus ETA.
    CUDA OOM candidates are reported and their cache is released.
    """
    if not candidates or any(batch < 1 for batch in candidates):
        raise ValueError("candidates must contain positive batch sizes")
    device = next(iter(model.layers[0].parameters())).device
    cuda = device.type == "cuda"
    rows = []
    for batch in candidates:
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        row = {"dim_batch": batch, "backward_passes": (model.d_model + batch - 1) // batch}
        try:
            jacobians, _, _ = jacobian_for_prompt(
                model, prompt, list(range(model.n_layers - 1)),
                dim_batch=batch, max_seq_len=max_seq_len,
            )
            del jacobians
            if cuda:
                torch.cuda.synchronize(device)
            row.update(status="ok", seconds=time.perf_counter() - started)
        except torch.cuda.OutOfMemoryError:
            row.update(status="OOM", seconds=None)
        if cuda:
            row["peak_allocated_GiB"] = torch.cuda.max_memory_allocated(device) / 2**30
            row["peak_reserved_GiB"] = torch.cuda.max_memory_reserved(device) / 2**30
            torch.cuda.empty_cache()
        rows.append(row)
        print(row, flush=True)
    return rows


def load_hf_model(
    model_id: str, *, device: str, dtype: torch.dtype
) -> LensModel:
    """Default worker loader using the original architecture-independent adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jlens.hf import from_hf

    hf_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return from_hf(hf_model, tokenizer, compile=False)


def _fit_shard(
    model_id: str,
    model_loader: Callable[..., LensModel],
    device: str,
    dtype: torch.dtype,
    prompts: list[str],
    shard_dir: str,
    fit_kwargs: dict,
    resume: bool,
) -> tuple[str, int]:
    # Limit CPU overhead when multiple CUDA workers are submitting kernels.
    torch.set_num_threads(1)
    directory = Path(shard_dir)
    output = directory / "lens.pt"
    if resume and output.is_file():
        return str(output), JacobianLens.load(str(output)).n_prompts
    directory.mkdir(parents=True, exist_ok=True)
    model = model_loader(model_id, device=device, dtype=dtype)
    lens = fit(
        model, prompts, checkpoint_path=str(directory / "fit_checkpoint.pt"),
        resume=resume, **fit_kwargs,
    )
    # Merge fp32 means, avoiding a lossy fp16 round trip for each shard.
    temporary = directory / "lens.tmp.pt"
    lens.save(str(temporary), dtype=torch.float32)
    temporary.replace(output)
    return str(output), lens.n_prompts


def fit_sharded(
    model_id: str,
    prompts: Sequence[str],
    *,
    checkpoint_dir: str | Path,
    devices: Sequence[str] = ("cuda:0", "cuda:0"),
    dtype: torch.dtype = torch.float32,
    model_loader: Callable[..., LensModel] = load_hf_model,
    resume: bool = True,
    **fit_kwargs,
) -> JacobianLens:
    """Fit one disjoint round-robin prompt shard per worker, then merge.

    ``devices`` defines both the worker count and placement. Repeating
    ``cuda:0`` uses separate processes on one GPU; no model instance or hook
    recorder is shared. ``model_loader`` must be an importable/picklable
    callable accepting ``(model_id, device=..., dtype=...)``; for the existing
    GPT-2 notebook, use ``GPT2LensModel.from_pretrained`` to preserve its BOS
    handling. The default uses ``jlens.from_hf``.

    Each shard has an independent resumable fit checkpoint and an fp32 lens
    file. A manifest rejects reuse with different prompts, model, dtype,
    loader, shard count, or fit options. ``merge`` weights by successfully
    fitted prompts, including uneven shard sizes and skipped short prompts.
    All shards must contain at least one valid fit prompt.

    Call from an importable script under a main guard, or from a notebook.
    This helper does not unload a model already held by the caller.
    """
    prompts = list(prompts)
    devices = list(devices)
    if not devices or len(prompts) < len(devices):
        raise ValueError("need at least one prompt per worker and at least one worker")
    if "checkpoint_path" in fit_kwargs:
        raise ValueError("use checkpoint_dir; each worker owns its checkpoint_path")
    fit_kwargs.setdefault("checkpoint_every", 5)
    manifest = {
        "model_id": model_id,
        "loader": f"{model_loader.__module__}.{model_loader.__qualname__}",
        "dtype": str(dtype),
        "n_shards": len(devices),
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "fit_kwargs": fit_kwargs,
    }
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    # Normalize tuples in fit options through JSON before comparing manifests.
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("shard manifest differs; use a new checkpoint_dir")
    else:
        if any(directory.glob("shard-*")):
            raise ValueError("shard directory has no manifest; use a new checkpoint_dir")
        temporary = directory / "manifest.tmp.json"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    output_paths = [None] * len(devices)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=context) as pool:
        futures = {
            pool.submit(
                _fit_shard, model_id, model_loader, device, dtype,
                prompts[index::len(devices)], str(directory / f"shard-{index:03}"),
                fit_kwargs, resume,
            ): index
            for index, device in enumerate(devices)
        }
        for future in as_completed(futures):
            index = futures[future]
            path, n_prompts = future.result()
            output_paths[index] = path
            print(f"Shard {index + 1}/{len(devices)} complete: {n_prompts} fitted prompts", flush=True)
    return JacobianLens.merge([JacobianLens.load(path) for path in output_paths])
