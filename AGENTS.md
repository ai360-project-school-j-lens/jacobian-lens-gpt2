# Repository Guidelines

## Project Structure & Module Organization

- `jlens/` contains the architecture-independent `LensModel` protocol, HuggingFace adapters, activation hooks, Jacobian fitting, evaluation, sharded fitting, and visualization.
- `gpt2/` provides the GPT-2 adapter and handwritten lens functions; `lens_eval.py` re-exports shared evaluation helpers.
- `notebooks/` groups walkthroughs, logit-lens examples, and dataset comparisons by lens. Use `notebooks/jacobian_lens/model_agnostic_lens_dataset.ipynb` for multi-model experiments.
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

Put reusable model-independent logic in `jlens/`. Prefer `LensModel` and `from_hf` over private architecture attributes. Clear notebook outputs and execution counts before committing.

## Testing Guidelines

Pytest runs function tests and unittest-based cases. Name files `test_*.py` and tests `test_*`. No coverage threshold is configured. Add regression tests for changed behavior using tiny CPU models rather than downloaded weights. Verify shared activations, final-model readout, tokenizer special-token handling, and shard merge/resume behavior when relevant.

## Commit & Pull Request Guidelines

History uses short descriptive subjects, such as `Switch notebooks to GPT-2 XL, add progress bars`; follow that style. Keep changes focused. PR descriptions should explain behavior, validation commands, and model/device assumptions. Link relevant issues and include plots or screenshots for visualization changes. Target this fork; the README describes upstream as a reference implementation that does not accept contributions.

## Experiment Configuration

Keep weights, `.pt` checkpoints, credentials, and downloaded corpora out of commits. Fit on independent corpora; merge only disjoint shards from the same model and fitting configuration. Record model ID, dtype, corpus mix, sequence length, and dimension batch size with experiment results.

## Other notes

NEVER use Russian in READMEs or comments.
