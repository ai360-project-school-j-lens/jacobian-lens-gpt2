"""Persistence and single-device calibration for the generic dataset notebook.

Sidecars detect accidental reuse, not changes to weight values or corpus revisions.
Legacy finals are never loaded without an explicit user-selected path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .hf import HFLensModel, _resolve_attr_path
from .lens import JacobianLens
from .protocol import LensModel


def persistence_root(use_google_drive: bool, local_root: Path) -> Path:
    """Mount Drive only in Colab; a failed mount must not silently fall back."""
    try:
        from google.colab import drive
    except ImportError:
        drive = None
    if use_google_drive and drive is not None:
        drive.mount("/content/drive")
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Google Drive is not mounted; refusing ephemeral storage.")
        return Path("/content/drive/MyDrive/jacobian-lens-runs")
    print("Using local persistence (ephemeral if this is a Colab runtime).")
    return Path(local_root).expanduser()


def runtime_metadata(
    model: LensModel, hf_model: torch.nn.Module, model_id: str,
    dtype: torch.dtype, device: str | torch.device,
) -> dict:
    """Validate HF adapter references and inspect metadata without reading weights.

    Managed HF provenance requires ``HFLensModel`` (including explicit custom
    ``Layout`` instances). A generic ``LensModel`` does not expose enough of its
    root/readout association to verify this contract.
    """
    if not isinstance(model, HFLensModel):
        raise RuntimeError("Managed HF provenance requires HFLensModel/from_hf; rebuild adapter and rerun configuration cell.")
    stale_adapter = (
        "Adapter and hf_model refer to different weights or stale modules; "
        "rebuild from_hf and rerun configuration cell."
    )
    if model._hf_model is not hf_model:
        raise RuntimeError(stale_adapter)
    # Resolve the adapter's actual layout, not a newly auto-detected layout.
    # Parameter membership alone misses replaced modules with shared parameters,
    # detached readouts, and entirely different roots sharing decoder blocks.
    try:
        text_module = _resolve_attr_path(hf_model, model.layout.path)
        components = (
            (model._text_module, text_module),
            (model.layers, getattr(text_module, model.layout.layers)),
            (model._embed_tokens, getattr(text_module, model.layout.embed)),
            (model._final_norm, getattr(text_module, model.layout.norm)),
            (model._lm_head, getattr(hf_model, model.layout.lm_head)),
        )
    except AttributeError as error:
        raise RuntimeError(stale_adapter) from error
    if any(retained is not current for retained, current in components):
        raise RuntimeError(stale_adapter)
    expected_device = torch.device(device)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())
    parameters = tuple(hf_model.parameters())
    parameter_ids = {id(p) for p in parameters}
    if any(id(p) not in parameter_ids for layer in model.layers for p in layer.parameters()):
        raise RuntimeError("Adapter and hf_model refer to different weights; rebuild from_hf and rerun configuration cell.")
    placements = {(str(p.dtype), str(p.device)) for p in parameters}
    if placements != {(str(dtype), str(expected_device))}:
        raise RuntimeError("Model dtype/device changed or is mixed; reload model and rerun configuration cell.")
    loaded_id = getattr(hf_model.config, "_name_or_path", model_id)
    if loaded_id != model_id:
        raise RuntimeError("Loaded model does not match MODEL_ID; reload model and rerun configuration cell.")
    tokenizer = model.tokenizer
    return {
        "model_id": model_id, "dtype": str(dtype).removeprefix("torch."),
        "device": str(expected_device), "mode": "single",
        "d_model": model.d_model, "n_layers": model.n_layers,
        "model_class": type(hf_model).__name__,
        "training": hf_model.training,
        "attention_implementation": getattr(hf_model.config, "_attn_implementation", None),
        "model_revision": getattr(hf_model.config, "_commit_hash", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "tokenizer_special_ids": list(getattr(tokenizer, "all_special_ids", [])),
        "tokenizer_add_bos": getattr(tokenizer, "add_bos_token", None),
        "torch_version": str(torch.__version__),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def legacy_final_candidates(repo_dir: Path, model_id: str,
                            drive_root: Path = Path("/content/drive/MyDrive")) -> list[Path]:
    """Known old final paths only (never checkpoints or automatic unverified loads)."""
    full = model_id.replace("/", "__")
    short = model_id.split("/")[-1]
    paths = [Path(repo_dir) / "lens_checkpoints" / full / "lens.pt"]
    old_drive = Path(drive_root) / "jacobian-lens"
    paths.extend([
        old_drive / f"{short}_jacobian_lens_dataset.pt",
        old_drive / f"{full}_jacobian_lens_dataset.pt",
        old_drive / full / "lens.pt",
        old_drive / "lens_checkpoints" / full / "lens.pt",
    ])
    return list(dict.fromkeys(path for path in paths if path.is_file()))


def save_json(path: Path, value: object) -> None:
    """Replace a JSON sidecar via a temporary file in the same directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_dim_batch(rows: list[dict], prompt_count: int, memory_budget_gib: float | None) -> int:
    """Choose the fastest summed timing safe on every representative prompt."""
    timings = pd.DataFrame(rows)
    if timings.empty:
        raise RuntimeError("No measured safe batch; lower candidates/sequence length or free memory.")
    timings["seconds"] = pd.to_numeric(timings.seconds, errors="coerce")
    safe = timings.status.eq("ok") & np.isfinite(timings.seconds) & timings.seconds.gt(0)
    if memory_budget_gib is not None:
        safe &= timings.peak_reserved_GiB.le(memory_budget_gib)
    candidates = timings[safe].groupby("dim_batch").seconds.agg(["count", "sum"])
    candidates = candidates[candidates["count"] == prompt_count]
    if candidates.empty:
        raise RuntimeError("No measured safe batch; lower candidates/sequence length or free memory.")
    return int(candidates["sum"].idxmin())


class DatasetFitRun:
    """Configuration-bound run directory; exact saved prompt order governs resume."""

    def __init__(self, root: Path, config: dict) -> None:
        self.config = json.loads(json.dumps(config))  # Snapshot nested mutable options.
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", config["model_id"])
        label = re.sub(r"[^A-Za-z0-9_.-]+", "-", config["run_label"])
        self.directory = Path(root) / (
            f"{slug}_{config['dtype']}_seq{config['max_seq_len']}_{label}_{digest}"
        )
        self.lens_path = self.directory / "lens.pt"
        self.checkpoint_path = self.directory / "fit_checkpoint.pt"
        self.prompts_path = self.directory / "ordered_prompts.json"
        self.manifest_path = self.directory / "manifest.json"
        self.benchmark_path = self.directory / "batch_selection.json"

    def print_paths(self) -> None:
        """Show the managed final, checkpoint, corpus, and provenance locations."""
        for label, path in (("Final lens", self.lens_path),
                            ("Resume checkpoint", self.checkpoint_path),
                            ("Ordered corpus", self.prompts_path),
                            ("Manifest", self.manifest_path),
                            ("Batch measurements", self.benchmark_path)):
            print(f"{label}: {path}")

    def _manifest(self, records):
        if not isinstance(records, list) or not records or any(
            not isinstance(row, dict) or not isinstance(row.get("source"), str)
            or not isinstance(row.get("text"), str) for row in records
        ):
            raise ValueError("Invalid saved ordered corpus.")
        digest = hashlib.sha256(
            json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        return {"config": self.config, "ordered_prompts_sha256": digest}

    def _verify(self, records):
        if json.loads(self.manifest_path.read_text(encoding="utf-8")) != self._manifest(records):
            raise ValueError("Run manifest/corpus mismatch; restore files or use a new RUN_LABEL.")

    def load_or_fit(
        self, model: LensModel, current_config: dict, *,
        load_prompts: Callable[..., pd.DataFrame], fit: Callable[..., JacobianLens],
        benchmark: Callable[..., list[dict]], display: Callable[..., object] = print,
        existing_path: str | Path | None = None, legacy_candidates: Sequence[Path] = (),
        allow_new_fit: bool = False,
    ) -> JacobianLens:
        """Validate before expensive work; explicit legacy override bypasses sidecars only."""
        if current_config != self.config:
            raise RuntimeError("Stale fit state; rerun configuration cell before managed load/resume/fit.")
        if existing_path is not None:
            path = Path(existing_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Requested existing final lens is missing: {path}")
            print(f"UNVERIFIED legacy lens: {path}; skipping downloads, benchmark, and fit.")
            lens = JacobianLens.load(str(path))
        elif self.lens_path.is_file():
            if not (self.prompts_path.is_file() and self.manifest_path.is_file()):
                raise RuntimeError("Managed final lens lacks corpus/manifest; restore sidecars or explicitly opt into EXISTING_LENS_PATH (unverified legacy).")
            records = json.loads(self.prompts_path.read_text(encoding="utf-8"))
            self._verify(records)
            print(f"Loading verified final lens: {self.lens_path}; skipping downloads, benchmark, and fit.")
            lens = JacobianLens.load(str(self.lens_path))
        else:
            found = [Path(path) for path in legacy_candidates if Path(path).is_file()]
            if found and not allow_new_fit:
                instructions = "\n".join(f"EXISTING_LENS_PATH = Path({str(p)!r})" for p in found)
                print("Legacy final candidate(s) found; not loading unverified and not starting a fit.\n"
                      "Select the matching FINAL lens in the configuration cell, then rerun it and the load/fit cell:\n"
                      + instructions)
                raise RuntimeError("Legacy final found. Opt into EXISTING_LENS_PATH above, or set ALLOW_NEW_FIT_WITH_LEGACY=True intentionally.")
            lens = self._fit(model, load_prompts, fit, benchmark, display)
        if lens.d_model != model.d_model or not set(range(model.n_layers - 1)).issubset(lens.source_layers):
            raise ValueError("Final lens width/layer coverage does not match this model.")
        return lens

    def _fit(self, model, load_prompts, fit, benchmark, display):
        config = self.config
        device = torch.device(config["device"])
        if torch.is_autocast_enabled(device.type) or torch.is_autocast_enabled("cpu"):
            raise RuntimeError("Run fitting outside autocast; no implicit reduced-precision fit.")
        batch = config["dim_batch_requested"]
        if batch is not None and (not isinstance(batch, int) or isinstance(batch, bool) or batch < 1):
            raise ValueError("DIM_BATCH must be None or a positive integer.")
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_path.exists() and not (self.prompts_path.is_file() and self.manifest_path.is_file()):
            raise RuntimeError("Checkpoint lacks corpus/manifest; restore sidecars or use a new RUN_LABEL.")
        if self.prompts_path.is_file():
            records = json.loads(self.prompts_path.read_text(encoding="utf-8"))
        else:
            if self.manifest_path.exists():
                raise RuntimeError("Manifest exists without corpus; restore it, do not resample.")
            records = load_prompts(config["fit_mix"], seed=config["seed"]).to_dict("records")
            self._manifest(records)  # Validate before persisting.
            save_json(self.prompts_path, records)
        if self.manifest_path.is_file():
            self._verify(records)
        else:
            save_json(self.manifest_path, self._manifest(records))
        prompts = pd.DataFrame(records)
        display(prompts.source.value_counts().to_frame("prompts"))
        if batch is None:
            lengths = [model.encode(text, max_length=config["max_seq_len"]).shape[1]
                       for text in prompts.text]
            valid = sorted((n, i) for i, n in enumerate(lengths) if n > config["skip_first"] + 1)
            if not valid:
                raise ValueError("No fit prompts have valid positions after truncation.")
            indices = list(dict.fromkeys([valid[-1][1], valid[len(valid) // 2][1]]))
            budget = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info(device)
                headroom = max(config["headroom_GiB"] * 2**30, config["headroom_fraction"] * total)
                budget = max(0, free + torch.cuda.memory_reserved(device) - headroom) / 2**30
                print(f"CUDA reserved-memory budget: {budget:.2f} GiB (headroom excluded)")
            rows = []
            for index in indices:
                print(f"Benchmark corpus index {index}: {lengths[index]} tokens after truncation")
                measurements = benchmark(model, prompts.text.iloc[index],
                                         candidates=config["candidates"], max_seq_len=config["max_seq_len"])
                rows.extend(dict(row, prompt_index=index) for row in measurements)
            display(pd.DataFrame(rows))
            batch = select_dim_batch(rows, len(indices), budget)
            save_json(self.benchmark_path, {"dim_batch": batch, "rows": rows,
                                           "memory_budget_GiB": budget, "device": str(device)})
        else:
            print("Manual batch: skipping calibration; memory safety has not been measured.")
            save_json(self.benchmark_path, {"dim_batch": batch, "manual": True})
        print(f"Single-worker fit: dtype={config['dtype']}, dim_batch={batch}, "
              f"backwards/prompt={(model.d_model + batch - 1) // batch}, prompts={len(prompts)}")
        print(f"Resume checkpoint exists: {self.checkpoint_path.is_file()}; {self.checkpoint_path}")
        lens = fit(model, prompts.text.tolist(), dim_batch=batch,
                   max_seq_len=config["max_seq_len"], skip_first=config["skip_first"],
                   checkpoint_path=str(self.checkpoint_path),
                   checkpoint_every=config["checkpoint_every"], resume=True)
        temporary = self.lens_path.with_suffix(".tmp.pt")
        lens.save(str(temporary), dtype=torch.float32)
        temporary.replace(self.lens_path)
        print(f"Saved final lens: {self.lens_path}")
        return lens


@contextmanager
def strict_identity_precision(device_type: str) -> Iterator[None]:
    """Disable TF32/autocast just for identity assertions, restoring even on failure."""
    previous_precision = torch.get_float32_matmul_precision()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with ExitStack() as stack:
            for kind in dict.fromkeys(["cpu", device_type]):
                stack.enter_context(torch.autocast(device_type=kind, enabled=False))
            yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)
        # Avoid setting the flag redundantly: some Torch versions map True to 'high'.
        if torch.backends.cuda.matmul.allow_tf32 != previous_tf32:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn
