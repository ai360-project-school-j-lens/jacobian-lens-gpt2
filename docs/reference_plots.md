# Reference-style plot adaptations

The **§8** section of
`notebooks/jacobian_lens/jacobian_logit_lens_dataset.ipynb` provides adaptations
of reference Figures 52/55/56. They are not numerical reproductions: model,
fitted lens, tokenizer, evaluation samples and held-out corpus differ. Original
sample subsets and symmetric-KL normalization are unknown. There is no tuned
lens, no reference-model workspace shading, and no claim that the supplied
held-out corpus matches a pretraining distribution.

## Running the notebook section

1. After the existing paired evaluation, run §8a. Figure 52 uses cached `words`;
   no refitting or extra model evaluation occurs. Read the displayed exclusion
   counts alongside the plot.
2. For Figures 55/56, edit §8b's `HELD_OUT_TEXTS` to a nonempty list of your own
   **independently held-out** strings and describe corpus/version, split and
   selection in `HELD_OUT_SOURCE`. A commented local-text-file example is
   provided. The default `None` prints guidance and skips all extra computation.
   No downloads, fitting-corpus reuse or task-prompt substitution occurs.
3. Run the evaluation cell once, then the plotting cell. Start with a few short
   passages. Default truncation is 128 tokens/text; position chunks are 4 and
   geometry vocabulary chunks 1024. All model blocks are evaluated. Reducing
   chunk size saves vocabulary memory; model activations and Jacobians still
   cost memory. These are extra forwards and geometry calculations, **not a
   refit**. Cache tables to regenerate plots without inference.
4. Record model ID/dtype, lens checkpoint/fitting provenance, sample selection,
   TF32 settings and truncation with any exported figures. The section prints
   run configuration and tables including actual token/text counts and geometry
   status. Do not infer statistical uncertainty from these descriptive curves.

## Reusable API (`jlens.reference_plots`)

```python
from jlens.reference_plots import (
    intermediate_rank_sweep, plot_intermediate_sweep,
    plot_distribution_summary, plot_lens_comparison,
)

sweep = intermediate_rank_sweep(words)
print(sweep.counts)
fig52, axes52 = plot_intermediate_sweep(sweep)
# metrics/geometry are generated explicitly using jlens.metrics; see lens_metrics.md.
fig55, axes55 = plot_distribution_summary(metrics.layers, n_layers=model.n_layers)
fig56, axes56 = plot_lens_comparison(metrics.pairs, geometry, n_layers=model.n_layers)
```

All plotting functions return `(Figure, axes)` without calling `show`; callers
can display or save figures. Helpers do not mutate inputs, perform inference,
load a model or access a corpus.

### Figure 52

- Panel order: multihop, multilingual, poetry / order-ops, association, typo.
- Cached ranks inherit `evaluate_paired()`'s `prompt.rstrip()` before
  tokenization: trailing whitespace is removed, which can change tokenization
  and the final-token readout position relative to the original prompt. This
  is an adaptation, not a whitespace-exact reproduction of the reference;
  the established evaluation semantics are unchanged.
- Intermediate word occurrences only, excluding `single_token=False` fallback
  rows. Counts are per dataset **and lens**, including items with no retained
  words, which are omitted rather than scored zero.
- At each k in `[1, 2, 5, 10, 20, 50, 100]`, a word hits if its minimum rank over
  all evaluated blocks (including shared final output) is at most k. Average
  hits per retained item, then average items equally. This is not the
  word-weighted `layer_hit_rate` aggregation and not generated-answer accuracy.
- Normalized log-k AUC = `sum((y[i]+y[i+1])/2 * log(k[i+1]/k[i])) /
  log(k[-1]/k[0])`. Per-panel legends show two decimals; black J-lens, red logit
  lens. Missing data is labeled, not silently plotted as zero.
- `IntermediateSweep.scores`: dataset, lens, k, score, auc.
  `.counts`: dataset, lens, n_words_total/used/excluded, n_items_total/used/excluded.

### Figures 55/56

Depth uses the actual model block count: `100 * layer / (n_layers - 1)`.
The first block output is 0 and final output 100; there is no embedding row.
For a one-block model the sole final output is 100. A subset of layers is
**not** stretched to fill the axis. The notebook requests every block.

55: KL(**model || lens**) and lens entropy in nats, model top-1 agreement in
**percent**. J-lens blue, logit lens purple. All distributions use the same input
positions; means weight valid input tokens equally, not texts equally.

56: one purple J/logit series in each panel. `same_layer_pairs` filters to
cross-lens, same-block pairs only (also works with legacy all-pairs tables).
It accepts either orientation but rejects mirrored duplicate rows instead of
silently averaging them. Symmetric KL is the **half-sum**, already computed by
`jlens.metrics`; it is not halved again, and is not Jensen–Shannon divergence.
Top-1 agreement is a **fraction** from 0 to 1.

Cosine compares corresponding vocabulary rows of `W_U` and `W_U @ J`, weighting
nonzero-vector pairs equally. This is **linear-head-only geometry**: bias,
normalization and softcap are excluded. It is not a global representation of
the nonlinear lens decoder. Unsupported/undefined geometry is unavailable,
not zero; inspect the geometry table's status, reason and vector counts.
The cosine axis matches the reference's **0 to 1** when all displayed values
are nonnegative (or unavailable). If any displayed cosine is negative, it
extends to **-1 to 1**, with an explicit plot-title disclosure of this adaptive
extension; scientifically real negative values are never silently clipped.
Probabilistic metrics do include the model adapter's full decoder. See
[lens_metrics.md](lens_metrics.md) for exact numerical/decoder conventions.
