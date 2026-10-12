"""Neuronpedia-compatible fit corpus and convergence-based early stopping.

The published Neuronpedia lenses (``neuronpedia/jacobian-lens`` on the Hub) are
fitted by ``fit_lens.py``: WikiText-103 records are concatenated and re-chunked
to ``max_chars`` characters, and the fit stops once the smoothed per-prompt
change of the running-mean Jacobian falls below ``stop_at_delta``. This module
reproduces both pieces so a lens fitted here follows the same recipe.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from jlens.fitting import FitProgress

CSV_COLUMNS = (
    "n_done",
    "prompt_idx",
    "seq_len",
    "n_valid_positions",
    "elapsed_s",
    "identity_distance",
    "mean_rel_change",
)


def chunk_texts(
    texts: Iterable[str],
    n_prompts: int,
    *,
    max_chars: int = 2000,
    min_chars: int = 200,
) -> list[str]:
    """Concatenate records and cut them into prompts of ``max_chars`` characters.

    Same rule as Neuronpedia ``fit_lens.load_prompts``: blank records and
    section headers (starting with ``=``) are dropped, records are joined with
    a space, and a final partial chunk is kept only if it has ``min_chars``.
    """
    prompts: list[str] = []
    buffer = ""
    for record in texts:
        text = str(record).strip()
        if not text or text.startswith("="):
            continue
        buffer += " " + text
        while len(buffer) > max_chars:
            prompts.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:]
            if len(prompts) >= n_prompts:
                return prompts
    if len(buffer.strip()) >= min_chars and len(prompts) < n_prompts:
        prompts.append(buffer.strip())
    return prompts


def load_chunked_prompts(
    dataset: str = "Salesforce/wikitext",
    config: str | None = "wikitext-103-raw-v1",
    *,
    split: str = "train",
    text_field: str = "text",
    n_prompts: int = 1000,
    max_chars: int = 2000,
) -> list[str]:
    """Stream a HF dataset and chunk it with :func:`chunk_texts`."""
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True)
    return chunk_texts(
        (record.get(text_field, "") for record in stream), n_prompts, max_chars=max_chars
    )


class ConvergenceTracker:
    """Per-prompt convergence log and early stop; pass :meth:`record` to ``fit``.

    The stop rule is Neuronpedia's: stop once at least ``min_prompts`` prompts
    are accumulated and the mean ``mean_rel_change`` of the last ``window``
    prompts is below ``stop_at_delta``. Rows go to a CSV with Neuronpedia's
    columns. On resume, rows past the checkpoint's ``n_done`` are dropped (they
    belong to prompts the resumed fit processes again) and the rest are
    replayed into the window, so a resumed fit stops at the same prompt.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        stop_at_delta: float | None = None,
        min_prompts: int = 100,
        window: int = 10,
        levels: Iterable[float] = (1e-2, 5e-3, 1e-3),
        resume_n_done: int = 0,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.stop_at_delta = stop_at_delta
        self.min_prompts = min_prompts
        self.window = max(1, window)
        self.levels = tuple(sorted(levels, reverse=True))
        self.crossed: dict[float, int] = {}
        self.stopped_at: int | None = None
        self._recent: deque[float] = deque(maxlen=self.window)
        rows = []
        if resume_n_done > 0 and self.csv_path.exists():
            with self.csv_path.open(newline="") as stream:
                rows = [row for row in csv.DictReader(stream)
                        if int(row["n_done"]) <= resume_n_done]
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            for row in rows:
                writer.writerow([row[column] for column in CSV_COLUMNS])
                self._note(int(row["n_done"]), float(row["mean_rel_change"]))

    def _note(self, n_done: int, value: float) -> None:
        if math.isnan(value):  # the first prompt has no previous mean
            return
        self._recent.append(value)
        for level in self.levels:
            if level not in self.crossed and value < level:
                self.crossed[level] = n_done

    def record(self, progress: FitProgress) -> bool:
        """Log one prompt; return ``True`` when the fit should stop."""
        with self.csv_path.open("a", newline="") as stream:
            csv.writer(stream).writerow([
                progress.n_done, progress.prompt_idx, progress.seq_len,
                progress.n_valid_positions, f"{progress.elapsed_s:.3f}",
                f"{progress.identity_distance:.6f}", f"{progress.mean_rel_change:.8f}",
            ])
        self._note(progress.n_done, progress.mean_rel_change)
        if (
            self.stop_at_delta is not None
            and progress.n_done >= self.min_prompts
            and len(self._recent) == self.window
            and sum(self._recent) / self.window < self.stop_at_delta
        ):
            self.stopped_at = progress.n_done
            return True
        return False
