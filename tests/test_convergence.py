import csv
import math

from jlens.convergence import ConvergenceTracker, chunk_texts
from jlens.fitting import FitProgress


def progress(n_done, delta):
    return FitProgress(prompt_idx=n_done - 1, n_prompts=100, n_done=n_done, seq_len=128,
                       n_valid_positions=111, elapsed_s=1.0, identity_distance=1.0,
                       mean_rel_change=delta)


def test_chunk_texts_matches_neuronpedia_rule():
    texts = ["", " = Title = ", "a" * 7, "b" * 7, "c" * 3]
    assert chunk_texts(texts, 10, max_chars=5, min_chars=3) == [
        "aaaa", "aaa b", "bbbbb", "b ccc",
    ]
    assert chunk_texts(texts, 2, max_chars=5) == ["aaaa", "aaa b"]
    assert chunk_texts(texts, 10, max_chars=5, min_chars=6) == ["aaaa", "aaa b", "bbbbb"]


def test_tracker_stops_on_smoothed_delta(tmp_path):
    tracker = ConvergenceTracker(tmp_path / "c.csv", stop_at_delta=0.1, min_prompts=4,
                                 window=2, levels=(0.5, 0.1))
    deltas = [math.nan, 0.6, 0.05, 0.05, 0.05]
    stops = [tracker.record(progress(n, d)) for n, d in enumerate(deltas, start=1)]
    assert stops == [False, False, False, True, True]
    assert tracker.stopped_at == 5 and tracker.crossed == {0.5: 3, 0.1: 3}


def test_tracker_resume_drops_rows_past_checkpoint(tmp_path):
    path = tmp_path / "c.csv"
    first = ConvergenceTracker(path, stop_at_delta=0.1, min_prompts=1, window=2)
    for n, d in [(1, math.nan), (2, 0.5), (3, 0.05), (4, 0.9)]:
        first.record(progress(n, d))
    resumed = ConvergenceTracker(path, stop_at_delta=0.1, min_prompts=1, window=2,
                                 resume_n_done=3)
    with path.open() as stream:
        assert [row["n_done"] for row in csv.DictReader(stream)] == ["1", "2", "3"]
    assert resumed.record(progress(4, 0.05))  # window (0.05, 0.05) from the replay
