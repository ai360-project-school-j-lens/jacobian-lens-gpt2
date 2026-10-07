"""Real spawned CPU workers: disjoint fitting, weighted merge, and resume."""

import logging
import multiprocessing
import time

import pytest
import torch

from jlens.fitting import fit
from jlens.sharded_fitting import benchmark_dim_batches, fit_sharded

from .tiny import TinyDecoder


def load_tiny_model(model_id, *, device, dtype):
    return TinyDecoder().to(device=device, dtype=dtype)


def test_shards_merge_like_serial_fit_and_resume(tmp_path, capsys):
    # Three valid prompts and one too short: successful shard counts are 2:1.
    prompts = ["abcdefghij " * 3, "x", "klmnopqrst " * 3, "uvwxyzabcd " * 3]
    kwargs = dict(dim_batch=4, max_seq_len=32, checkpoint_every=1)
    expected = fit(load_tiny_model("tiny", device="cpu", dtype=torch.float32), prompts, **kwargs)
    args = dict(
        checkpoint_dir=tmp_path, devices=["cpu", "cpu"],
        model_loader=load_tiny_model, **kwargs,
    )
    result = fit_sharded("tiny", prompts, **args)
    output = capsys.readouterr()
    assert "Starting 2 fit workers" in output.out
    assert "Merging completed shard lenses" in output.out
    assert "loading model" in output.err
    assert "jacobian backward" in output.err
    assert result.n_prompts == expected.n_prompts == 3
    for layer in result.source_layers:
        torch.testing.assert_close(result.jacobians[layer], expected.jacobians[layer], atol=1e-6, rtol=1e-6)
    paths = sorted(tmp_path.glob("shard-*/lens.pt"))
    assert len(paths) == 2
    timestamps = [path.stat().st_mtime_ns for path in paths]
    resumed = fit_sharded("tiny", prompts, **args)
    assert [path.stat().st_mtime_ns for path in paths] == timestamps
    for layer in result.source_layers:
        torch.testing.assert_close(resumed.jacobians[layer], result.jacobians[layer])
    with pytest.raises(ValueError, match="manifest differs"):
        fit_sharded("tiny", [*prompts[:-1], "changed prompt"], **args)


def test_benchmark_reports_real_pass_counts(capsys):
    previous_level = logging.getLogger("jlens.fitting").level
    rows = benchmark_dim_batches(TinyDecoder(), "abcdefghij " * 3, candidates=[2, 4], max_seq_len=32)
    assert [row["backward_passes"] for row in rows] == [4, 2]
    assert all(row["status"] == "ok" and row["seconds"] > 0 for row in rows)
    output = capsys.readouterr()
    assert output.out.index("Starting full-prompt Jacobian benchmark") < output.out.index("'status': 'ok'")
    assert "phase=forward" in output.err and "phase=backward" in output.err
    assert logging.getLogger("jlens.fitting").level == previous_level


def load_failing_model(model_id, *, device, dtype):
    if device == "broken":
        raise ValueError("worker fixture failed")
    time.sleep(30)
    return load_tiny_model(model_id, device="cpu", dtype=dtype)


def test_worker_errors_reach_parent_and_stop_other_workers(tmp_path):
    original_children = {child.pid for child in multiprocessing.active_children()}
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="worker fixture failed"):
        fit_sharded(
            "tiny", ["abcdefghij " * 3] * 2, checkpoint_dir=tmp_path,
            devices=["broken", "slow"], model_loader=load_failing_model,
        )
    assert time.perf_counter() - started < 20
    assert {child.pid for child in multiprocessing.active_children()} <= original_children


def test_empty_workers_or_shards_are_rejected(tmp_path):
    for prompts, devices in [([], ["cpu"]), (["long"], []), (["long"], ["cpu", "cpu"])]:
        with pytest.raises(ValueError, match="at least one"):
            fit_sharded("tiny", prompts, checkpoint_dir=tmp_path, devices=devices)
