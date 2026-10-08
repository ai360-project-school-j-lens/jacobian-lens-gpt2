# Notebook answer and retrieval summaries

`jlens.summaries` operates only on cached tables. No model passes, notebook edits,
tokenization, or changes to Figure 52 aggregation are involved.

```python
from jlens.summaries import (
    answer_counts, retrieval_summary, target_final_counts,
    plot_retrieval_summary, plot_target_final_counts,
)

# Pass all_words, not a table already filtered for token support or correctness.
display(answer_counts(items, datasets=dataset_names))
intermediates = retrieval_summary(
    all_words, items, kind="intermediate", k=10, layer_scope="inner",
    subsets=("all", "correct"), datasets=dataset_names,
)
display(intermediates)
fig, axes = plot_retrieval_summary(intermediates)

targets = retrieval_summary(all_words, items, kind="target", k=10)
fig, axes = plot_retrieval_summary(targets)

# This final-output view has no lens dimension: both lenses share this output.
final_targets = target_final_counts(all_words, items, k=10, datasets=dataset_names)
display(final_targets)
fig, axes = plot_target_final_counts(final_targets)
```

Plot functions return `(figure, axes)` without calling `show`; retrieval axes have
shape `(2,)`, and final-count axes have shape `(1, 1)`. Display the coverage tables
alongside the plots. N/A is not plotted as a zero-valued measurement.

## Input contract

- `items` requires `dataset`, `item`, `target`, and `model_correct`. `target` is
  missing for unannotated answers. `model_correct` is boolean for supported
  annotated answers, missing otherwise. True means the model's final argmax is
  an accepted whole-token answer spelling. A false value is **not** a substitute
  for unsupported. The optional `target_ids` must agree with this support status.
- Items may be repeated across `lens`. Model-level metadata must agree across
  copies (including correctness, target, prompt, model prediction, accepted target
  IDs and readout location where available); disagreements raise `ValueError`.
  Lens-specific readout/diagnostic arrays are deliberately not compared.
  Duplicate `(dataset, item, lens)` rows are rejected, not silently weighted twice.
- `all_words` requires `dataset`, `item`, `lens`, `kind`, `word`, `single_token`,
  and `ranks`. `single_token` is a boolean reporting **whole-token** support.
  Legacy intermediate/control prefix probes may carry ranks with
  `single_token=False`; these helpers exclude them. Unsupported ranks may be missing; they are never scored. Supported
  ranks must be nonempty positive integer arrays, in increasing block order,
  with the exact shared final-model output **last**. These helpers cannot infer
  layer order or repair a first-token fallback evaluator's ranks.
- Keep `role` if distinct annotations repeat a word. Word identity is
  `(dataset, item, lens, kind, role)`, or `word` in place of `role` when absent.
  Duplicate identities and word rows referencing unknown items are rejected.
- `datasets=None` preserves dataset order from `items`; an explicit sequence
  selects/orders datasets and includes empty datasets. There is no implicit
  correctness, in-prompt, or dataset-specific filter. Inputs are validated before
  selection, so malformed excluded rows are not silently accepted.

## `answer_counts(items, datasets=None)`

One row per dataset, counting each item once. Columns:

- `total`, `annotated`, `supported`, `correct`, `incorrect`, `unsupported`,
  `unannotated`, and `supported_accuracy`, plus `dataset`.
- `total = annotated + unannotated`;
  `annotated = supported + unsupported`;
  `supported = correct + incorrect`.
- `supported_accuracy = correct / supported`, NaN if supported is zero.
  Unsupported excludes unannotated items.

## `retrieval_summary(all_words, items, *, kind='intermediate', k=10, layer_scope='inner', subsets=('all', 'correct'), datasets=None)`

One row per dataset/lens/subset, with `kind`, `k`, `layer_scope` and `score`.
Lenses are the union of the lens names in both input tables (no hard-coded names).
`kind` may be `intermediate`, `target`, or `control`.

- `all`: **every item**, including items whose answers are unsupported,
  unannotated, or incorrect. A supported intermediate can still be scored.
- `correct`: exactly items with `model_correct == True`, shared across lenses.
- `inner`: best rank excluding the last entry (the shared final output).
  A one-entry rank array therefore has no eligible inner layers.
- `all`: best rank over all entries; `final`: only the last entry.
- For each eligible item, average the supported eligible words' rank hits
  (`rank <= k`), then average those item means equally. An item with five
  intermediates does not receive five times the weight of one with a single
  intermediate. Unsupported words are excluded, not treated as misses. A
  supported observed miss is zero; no eligible words/items produces NaN.

Coverage columns:

| Column | Denominator / meaning |
|---|---|
| `n_items_total` | Unique dataset items before subset selection |
| `n_items_subset` | Items in all/correct subset |
| `n_items_annotated` | Subset items with a word annotation of this kind/lens |
| `n_items_supported` | Subset items with at least one supported word |
| `n_items_used` | Subset items with at least one supported word and a selected layer |
| `n_items_excluded` | `n_items_subset - n_items_used` |
| `n_words_total` | All selected kind/lens word annotations in the subset |
| `n_words_supported` | Whole-token-supported annotations |
| `n_words_used` | Supported annotations with a selected layer |
| `n_words_excluded` | `n_words_total - n_words_used` |
| `item_coverage` | `n_items_used / n_items_subset` |
| `word_coverage` | `n_words_used / n_words_total` |

Zero-denominator coverage is NaN. An empty correct subset has zero counts and
NaN score/coverage. The default Figure 52 view remains
`jlens.reference_plots.intermediate_rank_sweep`, not this inner-layer summary.

## `target_final_counts(all_words, items, *, k=10, datasets=None)`

“Final token” here means the **annotated answer target at the evaluation readout
position, decoded from the final model layer**. It does not mean the last input
token or the model's own predicted token. No prediction-rank view is implemented.

Requires one target row per annotated item in at least one lens, at most one per
item/lens. Annotation and support must match `items`; available lens copies must
have identical final target ranks. Each item's target is counted once, regardless
of lens copies. Intermediate-layer ranks are allowed to differ.

Returns answer-count columns plus `k`, `hits`, `misses`, and `hit_rate`, with one
row for each dataset/cutoff in `{1, k}`. `hits` counts supported targets with final
deterministic lexical rank at most the cutoff; `misses = supported - hits`;
`hit_rate = hits / supported` (NaN without support). No-support hit counts are
zero counts, not an observed zero retrieval rate; plots mark these N/A.

A rank-1 hit is **not necessarily model-argmax correctness**. Ranks order lexical
scores descending, then token IDs ascending on exact ties; only one vocabulary
token has rank 1. With a dual readout, pre-softcap lexical rank 1 can differ from
the actual distribution argmax when finite-precision softcapping creates ties.
Use `answer_counts.supported_accuracy` for accepted-answer argmax accuracy.
See [readout ranking](readout_ranking.md) for tensor-only fallback semantics and
cache migration. These helpers do not repair old optimistic-tie or prefix-target
results; recompute those tables before comparing protocols.
