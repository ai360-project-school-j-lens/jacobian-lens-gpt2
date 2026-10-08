# Evaluations

Six prompt distributions used to evaluate lens quality (§methods-comparison). Each `{slug}.json` is prompts only.

## Conventions

Unless a section says otherwise:

- **Lens readout** — at each (layer, token position) the Jacobian lens
  returns a ranked list of vocabulary tokens.
- **Workspace band** — the contiguous mid-network layer range where
  workspace content is read; experiments report over this band, not
  individual layers.
- **Hit** — a target token is a *hit* if it appears at lens rank 1 at any
  (layer, position) in the band over the scored span.
- **Swap** — clamping a lens coordinate replaces one token's direction with
  another's at every band layer at the specified positions, then samples
  the continuation.
- Prompts that span multiple turns are given as
  `[{"role": "user"|"assistant", "content": ...}]`.

## Notebook answer diagnostics

The dataset protocols below score intermediates, not targets. The dataset notebooks
add **next-token answer diagnostics**: target ranks and model top-1 correctness use
exactly the same accepted complete single-token spellings (case/leading-space
variants, plus digit/word and operation synonyms for `order-ops`). A target with no
accepted whole-token spelling has null ranks and null correctness. A whitespace
or first-fragment prediction is not a completed multi-token answer.

Unsupported annotated targets are reported separately from unannotated items and
excluded from accuracy/rank denominators, not counted as wrong. Accuracy is
correct/supported; no supported answers means N/A. Legacy `pass_at_k` reports
`n_words`, `n_words_scored`, `n_items`, and `n_items_scored`; it excludes null ranks,
averages retained words within each item, then averages retained items equally.
Entirely unsupported groups retain NaN scores. Layer curves omit datasets with
no ranked words; consult the coverage tables rather than interpreting absence as
zero. Legacy intermediate/control probes still allow a first-token fallback;
strict decoded-spelling evaluation excludes those fallback probes too. These
policies do not measure generated multi-token answer accuracy.

`ReadoutCache` uses the same complete-target acceptance as direct evaluation.
`WordRanker` returns NaN for empty accepted-ID sets; `words_frame` restores null
rank rows and `item_scores` excludes them from denominators. Saved readout caches
now require scoring version 2; unversioned/older caches must be rebuilt, along
with their derived ranks/results. The convergence and residual-delta notebooks
use versioned cache filenames so obsolete answer-prefix results are not reused.

Ranks use descending lexical score, then ascending token ID on exact ties. With
HF dual readouts these are pre-softcap scores; correctness and predictions use
the actual model distribution. Finite-precision softcap saturation can therefore
make lexical rank 1 differ from model argmax, and the final lexical rank of the
model's predicted token need not be 1. Tensor-only callbacks instead rank the
distribution logits with the same deterministic tie rule. Recompute old
optimistic-tie/prefix-target result caches; see
[readout ranking and cache migration](../../docs/readout_ranking.md).

## lens-eval-multihop

[`lens-eval-multihop.json`](lens-eval-multihop.json)

Lens-quality eval (§methods-comparison). `items[*]` has `prompt` and `intermediates`. `target` defines the readout position only and is not itself scored. Readout is at a single position — the token immediately preceding `target` — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-multilingual

[`lens-eval-multilingual.json`](lens-eval-multilingual.json)

Lens-quality eval (§methods-comparison). `items[*]` has `prompt` and `intermediates`. `target` defines the readout position only and is not itself scored. Readout is at a single position — the token immediately preceding `target` — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-poetry

[`lens-eval-poetry.json`](lens-eval-poetry.json)

Lens-quality eval (§methods-comparison). `items[*]` has `prompt` and `intermediates`. Readout is at a single position — the last newline token (end of line 1 of the couplet) — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-order-ops

[`lens-eval-order-ops.json`](lens-eval-order-ops.json)

Lens-quality eval (§methods-comparison). Each intermediate is a key expanded to a synonym set (numbers → digit and word forms; operations → symbol and word forms); rank is the min over single-token synonyms at each layer. `items[*]` has `prompt` and `intermediates`. `target` defines the readout position only and is not itself scored. Readout is at a single position — the token immediately preceding `target` — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-association

[`lens-eval-association.json`](lens-eval-association.json)

Lens-quality eval (§methods-comparison). Each item is a short vignette that evokes a single concept (grief, Einstein, noir, ...) without ever naming it; `intermediates` holds that one concept word. `items[*]` has `prompt` and `intermediates`. Readout is at a single position — the final prompt token — the closing period — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.

## lens-eval-typo

[`lens-eval-typo.json`](lens-eval-typo.json)

Lens-quality eval (§methods-comparison). Each prompt is a sentence ending in a common misspelling; `intermediates` holds the single correctly-spelled word. `items[*]` has `prompt` and `intermediates`. Readout is at a single position — the final prompt token, i.e. the last tokenizer fragment of the misspelling — across all layers. Metric: pass@k = mean over items of the fraction of `intermediates` whose min-over-layers lens rank ≤ k.
