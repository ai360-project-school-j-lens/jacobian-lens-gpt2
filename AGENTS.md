# Repository Guidelines

## Project Structure & Module Organization

- `jlens/` contains the architecture-independent `LensModel` protocol, HuggingFace adapters, activation hooks, Jacobian fitting, evaluation, the readout-position cache (`readout_cache.py`), sharded fitting, and visualization.
- `gpt2/` provides the GPT-2 adapter and handwritten lens functions; `lens_eval.py` re-exports shared evaluation helpers.
- `notebooks/` groups walkthroughs, logit-lens examples, dataset comparisons, and experiments by lens; model-specific studies live in a per-model folder (`notebooks/qwen-9B/` for Qwen3.5-9B, strict and translated scoring). Use `notebooks/jacobian_lens/model_agnostic_lens_dataset.ipynb` for multi-model experiments.
- `tests/` contains offline tests and the small decoder fixture in `tiny.py`.
- `data/evaluations/` and `data/experiments/` hold JSON prompt sets and protocol documentation; `assets/` holds visualization resources.

## Build, Test, and Development Commands

Use Python 3.10 or newer. Dependencies and Ruff configuration live in `pyproject.toml`; `uv.lock` records resolved dependencies.

- `uv sync --extra dev` — create/update the local environment with development dependencies.
- `uv run --extra dev pytest -q` — run the test suite.
- `uv run --extra dev pytest tests/test_generic_evaluation.py -q` — check model-independent readouts and notebook analysis.
- `uv run --extra dev ruff check .` — lint Python code.
- `uv run jupyter lab` — open notebooks locally.
- `uv build` — build source and wheel distributions using setuptools.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, and `PascalCase` classes. Add type hints and concise docstrings to public functions. Ruff targets Python 3.10 with an 88-character line length; preserve surrounding style and avoid unrelated formatting changes.

Put reusable model-independent logic in `jlens/`. Prefer `LensModel` and `from_hf` over private architecture attributes.

Batch model passes whenever possible; never loop over prompts or settings one forward pass at a time. Group prompts by length, pad, and run them as a batch: with right padding, read each row at its last real token. Keep per-row state (bases, clean activations, answer masks) as batched tensors, move constants such as Jacobians to the device once, and avoid per-item GPU→CPU syncs (`.item()`, `int(tensor)`) inside the inner loop.

## Testing Guidelines

Pytest runs function tests and unittest-based cases. Name files `test_*.py` and tests `test_*`. No coverage threshold is configured. Add regression tests for changed behavior using tiny CPU models rather than downloaded weights. Verify shared activations, final-model readout, tokenizer special-token handling, and shard merge/resume behavior when relevant.

## Commit & Pull Request Guidelines

History uses short descriptive subjects, such as `Switch notebooks to GPT-2 XL, add progress bars`; follow that style. Keep changes focused. PR descriptions should explain behavior, validation commands, and model/device assumptions. Link relevant issues and include plots or screenshots for visualization changes. Target this fork; the README describes upstream as a reference implementation that does not accept contributions.

## Experiment Configuration

Keep weights, `.pt` checkpoints, credentials, and downloaded corpora out of commits. Fit on independent corpora; merge only disjoint shards from the same model and fitting configuration. Record model ID, dtype, corpus mix, sequence length, and dimension batch size with experiment results.

## Experiments

Deliver experiments as notebooks in `notebooks/<lens>/` (or a per-model folder such as `notebooks/qwen-9B/`), not as a separate README plus standalone scripts. Keep executed outputs (tables and plots) in committed notebooks so results are readable without rerunning. State the question and the answer at the top, describe the method, and end with conclusions. Cache expensive steps (fits, model passes) to files and skip them when the cache exists, so a notebook can be re-executed cheaply; keep the cached artifacts out of commits.

## Other notes

NEVER use Russian in READMEs, notebooks, or comments; all documentation is in English.
