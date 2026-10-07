"""Real spawned CPU workers: disjoint fitting, weighted merge, and resume."""

import pytest
import torch

from jlens.fitting import fit
from jlens.sharded_fitting import benchmark_dim_batches, fit_sharded

from .tiny import TinyDecoder


def load_tiny_model(model_id, *, device, dtype):
    return TinyDecoder().to(device=device, dtype=dtype)


def test_shards_merge_like_serial_fit_and_resume(tmp_path):
    # Three valid prompts and one too short: successful shard counts are 2:1.
    prompts = ["abcdefghij " * 3, "x", "klmnopqrst " * 3, "uvwxyzabcd " * 3]
    kwargs = dict(dim_batch=4, max_seq_len=32, checkpoint_every=1)
    expected = fit(load_tiny_model("tiny", device="cpu", dtype=torch.float32), prompts, **kwargs)
    args = dict(
        checkpoint_dir=tmp_path, devices=["cpu", "cpu"],
        model_loader=load_tiny_model, **kwargs,
    )
    result = fit_sharded("tiny", prompts, **args)
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


def test_benchmark_reports_real_pass_counts():
    rows = benchmark_dim_batches(TinyDecoder(), "abcdefghij " * 3, candidates=[2, 4], max_seq_len=32)
    assert [row["backward_passes"] for row in rows] == [4, 2]
    assert all(row["status"] == "ok" and row["seconds"] > 0 for row in rows)


def test_empty_workers_or_shards_are_rejected(tmp_path):
    for prompts, devices in [([], ["cpu"]), (["long"], []), (["long"], ["cpu", "cpu"])]:
        with pytest.raises(ValueError, match="at least one"):
            fit_sharded("tiny", prompts, checkpoint_dir=tmp_path, devices=devices)
