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
import logging
import multiprocessing
import queue
import time
import traceback
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from tqdm.auto import tqdm

from jlens.fitting import fit, jacobian_for_prompt
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@contextmanager
def _fit_logging(handler):
    """Observe fit records temporarily, including the per-Jacobian heartbeat."""
    logger = logging.getLogger("jlens.fitting")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


class _BenchmarkProgress(logging.Handler):
    def __init__(self, bar):
        super().__init__(logging.INFO)
        self.bar = bar

    def emit(self, record):
        message = str(record.msg)
        if message.startswith("  jacobian forward:"):
            self.bar.set_postfix(phase="forward", seq_len=record.args[0])
        elif message.startswith("  jacobian backward:"):
            done, _ = record.args
            self.bar.update(done - self.bar.n)
            self.bar.set_postfix(phase="backward")


class _WorkerProgress(logging.Handler):
    def __init__(self, events, shard):
        super().__init__(logging.INFO)
        self.events, self.shard = events, shard

    def emit(self, record):
        event = {"shard": self.shard, "phase": record.getMessage().strip()}
        message = str(record.msg)
        if message.startswith("  prompt ") or message.startswith("  resuming"):
            event["done"] = record.args[0]
        elif message.startswith("  skipping prompt"):
            event["done"] = record.args[0] + 1
        self.events.put(event)


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
        print(f"Starting full-prompt Jacobian benchmark: dim_batch={batch}", flush=True)
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        row = {"dim_batch": batch, "backward_passes": (model.d_model + batch - 1) // batch}
        try:
            with tqdm(total=row["backward_passes"], desc=f"benchmark dim_batch={batch}", unit="backward") as bar:
                with _fit_logging(_BenchmarkProgress(bar)):
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
    events,
    shard: int,
) -> tuple[str, int]:
    # Limit CPU overhead when multiple CUDA workers are submitting kernels.
    torch.set_num_threads(1)
    directory = Path(shard_dir)
    output = directory / "lens.pt"
    if resume and output.is_file():
        events.put({"shard": shard, "phase": "loading completed shard"})
        return str(output), JacobianLens.load(str(output)).n_prompts
    directory.mkdir(parents=True, exist_ok=True)
    events.put({"shard": shard, "phase": f"loading model on {device}"})
    model = model_loader(model_id, device=device, dtype=dtype)
    events.put({"shard": shard, "phase": "model loaded; starting fit"})
    with _fit_logging(_WorkerProgress(events, shard)):
        lens = fit(
            model, prompts, checkpoint_path=str(directory / "fit_checkpoint.pt"),
            resume=resume, **fit_kwargs,
        )
    events.put({"shard": shard, "phase": "saving shard lens"})
    # Merge fp32 means, avoiding a lossy fp16 round trip for each shard.
    temporary = directory / "lens.tmp.pt"
    lens.save(str(temporary), dtype=torch.float32)
    temporary.replace(output)
    return str(output), lens.n_prompts


def _fit_shard_process(*args) -> None:
    """Send results/errors over a pipe-backed queue to the notebook process."""
    events, shard = args[-2:]
    try:
        path, n_prompts = _fit_shard(*args)
        events.put({"shard": shard, "kind": "complete", "path": path, "n_prompts": n_prompts})
    except BaseException as error:
        events.put({"shard": shard, "kind": "error", "error": repr(error), "traceback": traceback.format_exc()})


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
    print(f"Starting {len(devices)} fit workers on {devices}; each worker loads its own model", flush=True)
    # A multiprocessing Queue uses pipes; no manager server/socket is needed.
    events = context.Queue()
    bars = [
        tqdm(total=len(prompts[index::len(devices)]), desc=f"fit shard {index + 1} ({device})",
             unit="prompt", position=index)
        for index, device in enumerate(devices)
    ]
    workers = []
    exited_without_result = {}
    try:
        for index, (device, bar) in enumerate(zip(devices, bars, strict=True)):
            bar.set_postfix(phase="starting worker")
            worker = context.Process(
                target=_fit_shard_process,
                args=(model_id, model_loader, device, dtype, prompts[index::len(devices)],
                      str(directory / f"shard-{index:03}"), fit_kwargs, resume, events, index),
                name=f"jlens-shard-{index + 1}",
            )
            worker.start()
            workers.append(worker)
        while any(path is None for path in output_paths):
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                event = None
            if event is not None:
                index = event["shard"]
                bar = bars[index]
                if event.get("kind") == "error":
                    raise RuntimeError(f"Shard {index + 1} failed: {event['error']}\n{event['traceback']}")
                if event.get("kind") == "complete":
                    output_paths[index] = event["path"]
                    bar.update(bar.total - bar.n)
                    bar.set_postfix(phase="complete", fitted_prompts=event["n_prompts"])
                    print(f"Shard {index + 1}/{len(devices)} complete: {event['n_prompts']} fitted prompts", flush=True)
                elif output_paths[index] is None:
                    if "done" in event:
                        bar.update(event["done"] - bar.n)
                    bar.set_postfix(phase=event["phase"])
            for index, worker in enumerate(workers):
                if output_paths[index] is not None:
                    continue
                if worker.exitcode is not None:
                    if worker.exitcode != 0:
                        raise RuntimeError(f"Shard {index + 1} exited with code {worker.exitcode}")
                    # Allow the final queue message to arrive before diagnosing
                    # a clean exit without a result.
                    exited_without_result.setdefault(index, time.perf_counter())
                    if time.perf_counter() - exited_without_result[index] > 1:
                        raise RuntimeError(f"Shard {index + 1} exited without a result")
                bars[index].refresh()
    finally:
        # Interrupts and worker errors must not wait for other whole shards.
        successful = all(path is not None for path in output_paths)
        for worker in workers:
            if successful:
                worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            if worker.is_alive():
                worker.kill()
                worker.join()
        for bar in bars:
            bar.close()
        events.close()
        events.join_thread()
    print("Merging completed shard lenses", flush=True)
    return JacobianLens.merge([JacobianLens.load(path) for path in output_paths])
