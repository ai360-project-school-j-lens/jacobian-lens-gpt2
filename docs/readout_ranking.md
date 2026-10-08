# Lexical ranking versus model distributions

`HFLensModel.unembed()` still returns the actual model's logits: final norm,
native-dtype head, and native-dtype softcap where configured. Neither weights
nor softcap arithmetic are promoted or changed. KL, entropy, model predictions,
correctness, agreement, and copy rate use these logits (distribution arithmetic
uses FP32). Held-out distribution evaluators keep this same convention.

Lexical ranking instead uses the head scores **before softcapping**. A softcap
is monotone in real arithmetic, but BF16/FP16/FP32 saturation can collapse many
unequal scores to the same maximum. Counting only strictly greater capped
logits previously gave all those tokens rank 1. Promoting already-capped logits
cannot recover their ordering; even FP32 softcapping saturates at large inputs.

## API and compatibility

- `jlens.LensReadout(logits, ranking_scores)` carries the two matching tensors.
- `from jlens.readout import readout; readout(model, residual)` uses the optional
  `model.unembed_readout()` capability. Otherwise both fields use `unembed()`.
  Existing `LensModel` implementations need no new method or keyword argument.
- `logit_lens`, `logit_lens_from_activations`, and `jacobian_lens` retain their
  historical tensor defaults. Use `return_readout=True` when passing them to
  `evaluate_readout`, `evaluate_all`, `lens_slice`, or `top_tokens_table`:

  ```python
  lens_fn = lambda text: logit_lens(model, text, return_readout=True)
  words, items = evaluate_readout(model, lens_fn, dataset, samples)
  ```

- Default `evaluate_paired` and `evaluate_paired_batched` use both spaces
  automatically. `compute_slice` displays lexical scores automatically.
- Custom `evaluate_paired(logit_readout=...)` callbacks should return a
  `LensReadout`. For backward compatibility, tensor callbacks still work, but
  **both lenses then rank distribution logits**, so identity comparisons remain
  meaningful. Known softcapped models issue a warning. This fallback prevents
  optimistic tie hits but cannot restore pre-softcap ordering. Update old
  handwritten notebook callbacks; do not interpret their fallback ranks as
  unsaturated lexical ranks.
- For a bounded custom baseline, use
  `evaluate_paired(layer_logit_readout=callback)`. The callback receives
  `(model, activations, layers=[layer])` and returns a `LensReadout` of shape
  `[1, sequence, vocabulary]`. It is called once per inner baseline layer;
  the final model readout is shared automatically. This option and the full
  `logit_readout` callback are mutually exclusive. The compatibility helper
  `evaluate_paired_layerwise(logit_readout=callback)` delegates to this same
  implementation, preserving its selected-layer callback contract.
- `WordRanker` accepts either a dual readout or a tensor. For cached residuals,
  pass `lambda h: readout(model, h)` to `rank_residuals`.
- `JacobianLens.apply` remains a distribution-logit API. Its tensor results do
  not encode the separate lexical scores.

Ranks now use a total order: descending score, then ascending token ID for
exact ties. `token_ranks`, cached and batched ranking, and lexical top-k displays
use this convention. Thus no more than k vocabulary tokens can have rank <= k.
This intentionally changes historical optimistic tie ranks, including on
non-softcapped models. Existing cached rank tables must be recomputed; a dtype
cast of saved capped logits is not a repair.

## Interpreting the tables

`words.ranks` measures lexical availability. `readout_top1`, `model_top1`,
`model_correct`, `readout_target_match`, agreement, and copy rate describe the
**actual finite-precision distribution**, not the lexical ordering. Softcap
ties can therefore make lexical target hit@1 differ from actual argmax
correctness even with identical accepted token IDs. Batched
`model_prediction_ranks` ranks the actual final argmax token in lexical scores;
its final-layer value is not necessarily 1. Lexical display top-1 can likewise
differ from `items.readout_top1`. Both lenses share the same final readout;
any fitted final-layer Jacobian is ignored in these evaluation/display helpers.

## Answer acceptance and coverage

Default `evaluate_readout`/`evaluate_paired` targets now require complete
single-token spellings, including the existing `order-ops` synonym expansion.
The same accepted IDs determine target ranks and actual-distribution
correctness. A multi-token-only target has null ranks and null correctness;
a whitespace or first-fragment prediction is not a completed answer. This
**changes historical target prefix acceptance**. Default intermediate/control
probes retain their legacy first-token fallback. Strict `DecodedSpellings`
lookup excludes prefix, special and whitespace-only tokens for every word kind;
custom spelling callbacks are responsible for that same contract.

Report supported, unsupported and unannotated counts. Unsupported targets are
excluded, not counted as wrong; no supported targets means N/A. These are
next-token diagnostics, not generated multi-token answer accuracy.

## Cache migration

Recompute saved rank/correctness tables made with optimistic ties or target
prefix acceptance. Cached accepted target IDs must also be rebuilt; recomputing
ranks with stale prefix IDs is insufficient. `ReadoutCache` serialization
version 2 rejects older/unversioned bundles because their target IDs may be
prefixes; rebuild those bundles and derived results. Residual tensors and fitted
Jacobians are not themselves changed by these scoring rules, provided their
model, prompt encoding and readout-position provenance still matches.

Cache keys must include the rank space (pre-softcap or tensor-only distribution
fallback), descending-score/ascending-ID tie policy, answer acceptance policy,
and source hashes including `readout.py` and the applicable evaluator/scorer.
A source-keyed cache invalidates automatically when those dependencies change;
a fixed numerical scoring version must instead be bumped. Table-schema or
whole-token-acceptance versions alone do not identify the ranking convention.
Do not relabel old numerical results with new protocol metadata.

## Limitations and cost

Pre-softcap scores retain native head precision; head-rounding ties are broken
by ID, not reconstructed using an FP32 head. Tensor-only callbacks and adapters
without the optional capability cannot recover lost ordering. Automatic
warnings recognize the HF adapter's known softcap, not arbitrary custom
transforms. Custom transforms that are not monotone need their own explicitly
defined lexical score convention.

Dual readouts add a vocabulary-sized score buffer for softcapped models.
Default `evaluate_paired` streams layers: only the final reference and one
current dual readout are retained, with layer-local distribution workspace.
Final distribution statistics are computed once per lens, not once per layer;
no two-layer stacks or partial-table concatenation are needed. The model-agnostic
and failed-Gemma dataset notebooks use this default production path. Their
handwritten identity checks use the selected-layer callback; their final-model
checks decode only the final layer. Historical notebook outputs remain stale.
Custom callbacks can still return full per-layer tensors, but evaluation does
not clone or modify them (including aliased/expanded views); the final reference
is substituted when reading rather than written into callback-owned storage.
Standalone `logit_lens`/`jacobian_lens` still return full tensors by contract.
For larger jobs, prefer `evaluate_paired_batched`, whose vocabulary workspace is
also bounded by the configured position chunk size. Activation and Jacobian
storage is separate from these readout bounds.

`top_token_ids` returns compact, independently owned top-k indices rather than
a view retaining a vocabulary permutation. It sorts at most 4096 vocabulary
entries per row chunk and merges at most 2k candidates, breaking ties by ID.
For k covering at least half the vocabulary, it instead sorts a row chunk once
(the permutation width is then at most 2k), avoiding repeated large merges.
`selected_token_ranks` bounds row, target, and vocabulary comparison dimensions;
its optional `row_indices` gathers inside vocabulary chunks instead of copying
whole rows for every word. Batched and cached rankers use that path. Displays
rank only tracked tokens, not a full vocabulary permutation and inverse.
Sorting/comparison work still scales with vocabulary size (and requested target
count), but temporary storage does not grow with all positions or targets.
Stable lexical selection costs more than an unordered `topk`. These are lexical
measurement changes, not a claim to improve model generation or lens fitting.
