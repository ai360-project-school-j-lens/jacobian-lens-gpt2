# Strict batched paired evaluation

`jlens.batched_evaluation.evaluate_paired_batched` is an opt-in evaluator for
raw-text general/HuggingFace/Qwen experiments. It preserves the paired table
schema while requiring strict whole-token scoring. All evaluators now use
[deterministic ranks](readout_ranking.md); the legacy evaluators also require
complete target spellings, while retaining intermediate/control prefix probes.

## Notebook API

```python
from jlens.batched_evaluation import evaluate_paired_batched
from jlens.strict_scoring import DecodedSpellings, ExplicitPromptModel

# hf_model and tokenizer must match the fitted lens's model ID/revision.
# from_hf(..., force_bos=False) leaves BOS selection to this explicit wrapper.
model = ExplicitPromptModel(adapter, bos_policy="none")
spellings = DecodedSpellings(
    tokenizer, vocab_size=hf_model.get_output_embeddings().weight.shape[0],
)
all_words, items = evaluate_paired_batched(
    model, fitted_lens, evals,
    spelling_lookup=spellings,
    preserve_prompt_whitespace=True,
    batch_size=8,
    max_batch_tokens=2048,
    max_seq_len=512,
    position_chunk_size=8,
    rank_chunk_size=8,
    batching="auto",
    progress=True,
)
print(items.attrs["evaluation"])
```

Full signature:

```python
evaluate_paired_batched(
    model, lens, evals, desc="batched paired lenses", *,
    spelling_lookup, bos_policy=None, preserve_prompt_whitespace=True,
    batch_size=8, max_batch_tokens=2048, max_seq_len=512,
    position_chunk_size=8, rank_chunk_size=8, batching="auto", progress=True,
) -> tuple[pandas.DataFrame, pandas.DataFrame]
```

- `evals` is the existing `{dataset: [item, ...]}` schema. Each item has `name`,
  `prompt`, `intermediates`, and optionally `target`. Original dataset/item/lens
  row ordering is restored after batching. Lens names are `logit lens`, `J-lens`.
- Supply an `ExplicitPromptModel`, or pass `bos_policy="none"`, `"prepend"`, or
  `"tokenizer"` to wrap another adapter. Conflicting policies fail. BOS choices
  must also be used by native-HF sanity checks and interactive views.
- `spelling_lookup` is required. Use `DecodedSpellings`, or a callable with the
  same strict whole-token contract. It is **not** a glossary or first-token
  fallback. The evaluator trusts custom callables' lexical semantics.
- All whitespace is preserved; `preserve_prompt_whitespace=False` fails. No
  chat template, generation, or automatic truncation occurs. Overlength inputs
  fail, including when a single input exceeds `max_batch_tokens`.
- There is no `logit_readout` callback accepting a full layer/vocabulary tensor:
  the baseline is the adapter's complete `unembed` of each raw block output.
  Distribution logits include its final normalization, head, bias and any logit
  softcap. Lexical ranks use `unembed_readout` scores when available (pre-softcap
  for the HF adapter); otherwise they use distribution logits.

## Batching and adapter safety

Prompts are tokenized on CPU through `ExplicitPromptModel.encode_ids`, sorted
by **encoded length**, and grouped across datasets. Controls still come from
the original dataset ordering. `max_batch_tokens` bounds
`batch_size_actual * longest_sequence_length`, including BOS and padding;
`None` removes that bound. `batch_size` remains a hard row limit.

- `auto`: masked right-padding when the adapter advertises
  `supports_attention_mask=True`; exact equal-length groups otherwise.
- `padded`: require that capability or raise, never silently discard a mask.
- `equal_length`: no padding, no attention-mask keyword. Common token lengths
  still share forwards; unique lengths necessarily use singleton batches.

`HFLensModel` advertises mask support only when the bare text decoder's
`forward` signature explicitly contains `attention_mask`. Its new optional
`forward(input_ids, attention_mask=...)` forwards that mask with
`use_cache=False`. A bare `**kwargs` does not qualify. Normal existing
`forward(input_ids)` calls are unchanged. This supports native HF text-only
and multimodal-wrapper text decoders, including Qwen3.5 hybrid linear/full
attention, without substituting a GPT-2-specific forward path.

Custom adapters must support independent batched causal rows, eval-mode
blocks, `[batch, sequence, hidden]` block outputs, and arbitrary position
batches in `unembed`. To opt into padding they must explicitly advertise and
correctly implement the mask contract. Architectures with cross-row state,
persistent recurrent caches, or incompatible output shapes are not supported;
do not advertise a capability merely because `forward` accepts `**kwargs`.
CPU/CUDA compute only. Unknown mask APIs use the equal-length baseline.

Every row uses its real token positions, never the last padded column. Poetry
uses the last **real** token whose decoded text contains a newline, falling
back to the last real token. BOS, specials and padding are excluded from
behavioral metric denominators; a special readout position can still be
ranked if specified by the dataset rule.

## Scoring and output schema

The existing word columns are unchanged:

```text
dataset item kind word role single_token in_prompt
ranks best_rank best_layer lens
```

`accepted_ids` additionally stores the sorted tuple of exact accepted IDs.
`ranks` is a 1-based integer array over all blocks, minimum over those IDs.
Ranks order lexical scores descending, breaking exact ties by ascending token
ID. There is exactly one vocabulary token at rank 1, but pre-softcap lexical
rank 1 can differ from the actual distribution argmax after saturation.
An empty accepted set has `single_token=False` and null
`ranks/best_rank/best_layer`, never a fallback rank. Null scalars may appear as
NaN in pandas. Unsupported words remain in `all_words` for coverage counts.

The existing item columns are unchanged:

```text
dataset item prompt target readout_token model_top1 model_correct
readout_top1 agreement copy_rate kl_to_final lens
```

Additional columns:

| Column | Meaning |
| --- | --- |
| `model_top1_id` | Independent final model argmax at the dataset readout position |
| `model_prediction_ranks` | Per-block deterministic lexical rank of that exact final distribution argmax ID; final rank need not be 1 after softcapping |
| `readout_top1_ids` | Actual distribution argmax ID array, one per block (not lexical top-1) |
| `target_ids` | Same accepted IDs used for the target word's ranks |
| `readout_target_match` | Boolean array testing each argmax against `target_ids`; null if unsupported/unannotated |
| `n_tokens` | Non-special real positions contributing to behavioral metrics |
| `n_prompt_tokens` | Encoded length including any BOS/specials, excluding padding |
| `readout_position` | Zero-based position in that encoded prompt |

`agreement`, `copy_rate`, and `kl_to_final` are per-item/per-layer means over
non-special real input positions, including the last. KL is model-to-lens in
nats. Final model logits are computed from the untransported final block;
any fitted final Jacobian is ignored. Both lenses share final ranks, argmaxes,
correctness and metrics exactly. Final agreement is 1 and final KL is 0.

`model_correct` and `readout_target_match` use exact token-ID membership, not
stripped display strings, translations or rank-1 shortcuts. They use the same
order-ops synonym expansion as target ranks. Empty targets are unscorable,
not incorrect. These are lexical next-token matches, not generated-answer
accuracy or multi-token completion likelihood.

`model_prediction_ranks` does not use the annotated target, spelling lookup,
or glossary. It tracks where the **model's own final predicted token** ranks
through the layers, including unannotated/unsupported targets. It is distinct
from target ranks, argmax agreement and averaged behavioral metrics. There is
no additional forward; vocabulary comparisons use `rank_chunk_size` bounds.

## Counts, all-versus-correct subsets, and Figure 52

Model counts must not double-count the two lens rows:

```python
import numpy as np

model_items = items.drop_duplicates(["dataset", "item"])
counts = model_items.assign(
    annotated=lambda f: f.target.notna(),
    supported=lambda f: f.model_correct.notna(),
    correct=lambda f: f.model_correct.eq(True),
).groupby("dataset").agg(
    total=("item", "size"), annotated=("annotated", "sum"),
    supported=("supported", "sum"), correct=("correct", "sum"),
)
counts["incorrect"] = counts.supported - counts.correct
counts["unsupported"] = counts.annotated - counts.supported
counts["unannotated"] = counts.total - counts.annotated
counts["final_token_match_supported"] = (
    counts.correct / counts.supported.replace(0, np.nan)
)
```

For per-layer target argmax hit rates, average `readout_target_match` arrays
only over rows where that array is non-null, separately by lens/dataset. Their
last entries equal the supported-target final token rate above. Do not replace
these with `target.ranks[-1] <= 1`: finite-precision softcap ties can make
pre-softcap lexical ordering differ from the distribution argmax.

For intermediate all-versus-model-correct plots, filter on **shared final
model correctness**, not the intermediate lens prediction. Use unique
`dataset/item` keys and preserve each lens's separate word rows:

```python
from jlens.reference_plots import intermediate_rank_sweep

correct_keys = model_items.loc[
    model_items.model_correct.eq(True), ["dataset", "item"]
]
correct_words = all_words.merge(
    correct_keys, on=["dataset", "item"], validate="many_to_one",
)
all_sweep = intermediate_rank_sweep(all_words)
correct_sweep = intermediate_rank_sweep(correct_words)
```

These are the existing Figure 52 `IntermediateSweep(scores, counts)` schemas:
`scores` has `dataset/lens/k/score/auc`; counts include total, used and excluded
words/items for each lens/dataset. The helper selects supported intermediates,
uses best rank over **all** blocks, averages word hit fractions within each
item, then equally across eligible items. Use the unfiltered `all_words` as
input so unsupported-word counts are retained. "All" includes items with no
supported target; "correct" does not. Empty correct subsets do not imply zero
scores; reindex missing plot panels as unavailable. For inner-only plots,
explicitly exclude the final rank column rather than relabeling this sweep.
The shared legacy rank summaries exclude null ranks and expose coverage;
custom consumers must likewise handle unsupported rows explicitly.

Glossary handling stays in the notebook's display layer:
`build_page(..., alt_token=gloss)` and `compute_slice(..., mask_display=True)`.
Keep original tokenizer strings and full-vocabulary ranks. Never re-encode a
gloss or pass gloss translations to `spelling_lookup`.

## Batched held-out distributions (Figures 55/56)

```python
from jlens.batched_evaluation import evaluate_distributions_batched

metrics = evaluate_distributions_batched(
    model, fitted_lens, held_out_texts,
    batch_size=8, max_batch_tokens=2048, max_seq_len=512,
    position_chunk_size=8, batching="auto", progress=True,
)
layer_table, pair_table = metrics.layers, metrics.pairs
print(layer_table.attrs["evaluation"])
```

The signature above includes every optional argument and its default.
`max_batch_tokens=None` removes the padded-token budget, not the row bound.
Returns the existing `jlens.metrics.DistributionMetrics` object and exact
[layer/pair column schemas](lens_metrics.md#exact-layers-table-contract):

- All blocks; `2 * n_layers` layer rows (logit lens, then J-lens).
- Exactly `n_layers` **same-layer, cross-lens** pair rows. No cross-layer
  Cartesian product or self pairs.
- Layer metrics: `KL(model || lens)`, entropy in nats, raw-logit top-1 agreement.
- Pair metrics: half-sum symmetric KL and raw-logit top-1 agreement.
- Strict token weighting over all non-special real input positions, including
  the last. Exclude tokenizer special IDs, BOS/EOS/pad IDs, and synthetic
  right-padding by true lengths. Do not filter the output vocabulary.
- `n_texts` counts all supplied texts; `n_texts_used` counts only those with
  valid positions; `n_tokens` is the shared denominator. Special-only texts
  do not cause forwards. No valid positions means `ValueError`.
- The untransported final readout is reused for both lenses: final KL and
  pair symmetric KL are exactly 0; agreements exactly 1. Final J is ignored.

Supply independent held-out strings, not the fitting corpus or an implicit
replacement by task prompts. The distribution evaluator has no targets,
spelling lookup, glossary, rank scoring or chat template. It does not fit or
load a model. Original whitespace is preserved.

**Encoding policy:** an `ExplicitPromptModel` uses `encode_ids`, retaining its
chosen BOS policy and rejecting empty/overlength encodings exactly as its
serial `encode` does. A bare `HFLensModel` uses CPU `encode_ids` with its normal
tokenizer specials and per-text truncation to `max_seq_len`. Therefore use the
same wrapper in serial/batched comparisons. Other adapters may expose an
identically behaving CPU `encode_ids`; otherwise the evaluator falls back to
`encode` and copies each input ID vector to CPU for length grouping. It never
silently changes an explicit policy or packs texts together.

The same `auto/padded/equal_length` capability checks and masked right-padding
implementation serve both task and held-out evaluators. Both returned tables
record `attrs["evaluation"]`: resolved mode, `n_model_passes`, actual batch sizes,
requested batching/chunk limits, BOS policy and corpus counts. Save that
metadata separately if the serialization format does not retain DataFrame attrs.

Inputs are tokenized/materialized for length sorting. Only final and current
layer distributions coexist, `O(position_chunk_size * vocabulary)` workspace;
no all-layer or corpus logits are retained. One fp32 Jacobian per inner
layer/device is cached for the call. Full padded-batch block activations remain
resident. Device fp64 reductions transfer to CPU once per batch, not per token
or readout; CPU/CUDA only, eval-mode blocks required. Finiteness is checked per
batch. Caller TF32 settings and the adapter's final norm/head/softcap remain
in effect. Numerical near-ties can change with batched GEMMs.

An offline smoke compared this API with serial `evaluate_distributions` on
random tiny native HF GPT-2, Llama and hybrid Qwen3.5, using both bare HF and
explicit BOS wrappers. Five usable texts plus two special-only texts used two
masked forwards. Masked/equal-length modes, chunk sizes 1/3/64, missing pad IDs,
real special/pad tokens, exact final identity and token/text counts agreed.
A protocol-only tiny decoder also exercised equal-length/encoding fallback.
No weights, GPU measurements, formal tests or notebook edits were needed.

## Performance, cache keys and validation

One model forward is made per batch, shared by both lenses. Both result frames
record `attrs["evaluation"]` with resolved batching mode, actual batch sizes,
number of model passes, item count, BOS and batching/chunk settings. Include
this implementation, `readout.py`, `evaluation.py`, `strict_scoring.py`, `hf.py`,
model/lens revisions, rank space/tie policy and answer acceptance settings in
notebook cache provenance. Invalidate pre-change rank/correctness tables; an
unchanged table schema does not imply unchanged numerical scoring. Inference caches are caller
managed; this API never fits, downloads, or writes checkpoints.

Readouts stream over position chunks and layers. Only the final distribution
and current readout coexist; vocabulary workspace scales as
`O(max(position_chunk_size, rank_chunk_size) * vocabulary_size)`, not with the
number of layers or corpus prompts. Ranking selects the highest accepted-token
lexical score (smallest ID on ties), then counts greater scores and equal scores
with smaller IDs in bounded word-row chunks. It never
sorts or retains every layer's full vocabulary logits.

Full batch block activations are still retained: approximately
`layers * batch * padded_length * hidden_width * activation_bytes`. Every
inner Jacobian is cached once per activation device in fp32 for this call:
approximately `inner_layers * hidden_width**2 * 4` bytes. These can dominate
memory on large Qwen models; lower the prompt batch/token budget for
activations and the position/rank chunk sizes for vocabulary workspace.
There is no CPU-J streaming/offload option in this API. This is a bounded
**workspace** claim, not a layer-independent bound on total memory. The
Jacobian cache has at most one entry per inner layer/activation device and
is reused across batches; it does not accumulate corpus-dependent tensors.
Residuals are position-gathered before fp32 conversion and transport, so
transient transport storage is `O(position_chunk_size * hidden_width)`.
Transported activations are not retained across layers or chunks; task logits
and held-out current-layer distribution pairs are explicitly released after
each layer. No all-layer fp32 activation copy is constructed.

No per-item GPU scalar readback occurs in the readout loop. Ranks, argmax IDs
and reduced metrics transfer to CPU once per batch; Python decoding and row
construction happen afterward. Nonfinite logits/metrics fail with a batch
level check. Autocast is disabled; transport uses fp32, distributions fp32 and
aggregate sums fp64. Model parameters are never cast or mutated: only gathered
residuals, Jacobian copies and distribution tensors are converted. The HF
adapter casts each readout residual to its existing head dtype, not the head
to fp32. Adapter head dtype and caller TF32 settings remain in effect. Batched GEMMs can differ slightly from singleton GEMMs, especially
bf16/TF32; near-tie rank/argmax changes are possible, not a semantic change.

A small offline CPU smoke compared strict legacy and batched results using
random native HF GPT-2, Llama and Qwen3.5 text models (the latter with one
linear-attention and one full-attention block). It covered masked padding,
equal-length fallback, nonidentity inner/final Jacobians, unsupported words,
order-ops synonyms, exact whitespace, explicit BOS, poetry positions, and
native final-logit argmaxes. Five prompts used two padded model passes versus
five singleton passes (four with exact-length grouping). No weights were
downloaded. No large suite, GPU/large-Qwen benchmarks, custom fused recurrent
kernels, quantized weights or offloaded model execution were validated.

The existing 91,695-entry `assets/qwen_gloss.json.gz` was inspected as display
metadata and rejected when supplied as a scoring lookup (it is not callable).
Notebook wiring keeps it solely in `alt_token=gloss`; no evaluator reads it.
Custom spelling callables are still trusted and must not incorporate glossary
translations. The one exception is the separate, opt-in cross-lingual protocol
`jlens.translated_scoring.TranslatedSpellings`, whose results are reported next
to strict results, never in their place (see
`notebooks/jacobian_lens_prefitted/qwen35_9b_lens_translated.ipynb`). The original non-softcapped smoke checked prediction ranks against
direct logit comparisons and observed final rank 1; that observation is not a
general guarantee. The softcap regression in `tests/test_readout_ranking.py`
checks deterministic pre-softcap ranks and a final `model_prediction_ranks`
value greater than 1 while distribution predictions remain unchanged.
