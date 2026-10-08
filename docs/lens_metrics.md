# Held-out lens metrics and linear-head geometry

Recommended batched API for new Figure 55/56-style notebook analyses:

```python
from jlens.batched_evaluation import evaluate_distributions_batched
from jlens.metrics import lens_vector_geometry

metrics = evaluate_distributions_batched(
    model, fitted_lens, held_out_texts,
    batch_size=8, max_batch_tokens=2048, max_seq_len=128,
    position_chunk_size=8, batching="auto", progress=True,
)
layer_table = metrics.layers
pair_table = metrics.pairs
geometry_table = lens_vector_geometry(
    model, fitted_lens, vocab_chunk_size=4096, progress=True,
)
```

These support **adaptations** of reference Figures 55/56, not an exact
reproduction of their corpus, model, or experimental protocol. Plotting is
intentionally outside this module. No model loading or text downloading occurs.
Record the model ID, model/lens dtype, TF32 settings, corpus, truncation, and
fitting configuration with results. Use independently held-out texts, not the
fitting corpus or an implicitly substituted task-prompt set.

## Batched distribution API

`jlens.batched_evaluation.evaluate_distributions_batched` returns the exact
`DistributionMetrics.layers` / `.pairs` schemas documented below, for all
blocks and same-layer cross-lens pairs only. It shares the task evaluator's
length grouping, attention-mask handling and cached Jacobians. Non-special,
non-padding positions receive strict token weights; full-vocabulary readouts
are chunked by position and streamed by layer. The last real position counts.
One shared forward is performed per batch, not per text. See
[batched evaluation](batched_evaluation.md#batched-held-out-distributions-figures-5556)
for the signature, adapter/encoding policies, memory limits and provenance.
Use it for new experiments; the serial API below is retained for compatibility
and reference comparisons, not as the recommended corpus execution path.

## Serial reference distribution API

```python
evaluate_distributions(
    model: LensModel,
    lens: JacobianLens,
    texts: Iterable[str],
    *,
    layers: Sequence[int] | None = None,
    max_seq_len: int = 512,
    position_chunk_size: int = 8,
    pairwise: bool = True,
    progress: bool = True,
    desc: str = "held-out lens metrics",
) -> DistributionMetrics
```

`DistributionMetrics` is a frozen dataclass with two mutable pandas DataFrames,
`layers` and `pairs`. It contains no texts, logits, or model references.

### Decoding and position conventions

- Layer indices are **zero-based block outputs, before final normalization**.
- `layers=None` requests all blocks. Otherwise elements must be Python integers
  in the model's block range; booleans and nonintegers raise `ValueError` before
  deduplication/sorting. This also applies to geometry's `layers` argument.
  The final block is **always added**. Every requested inner layer must have a
  fitted Jacobian; missing matrices, wrong dimensions, and nonfinite matrices
  raise `ValueError`. An arbitrary stored final-layer matrix is ignored.
- One forward pass per usable text captures the same activations for both
  readouts. The caller must provide an eval-mode, deterministic `LensModel`.
- `logit lens`: `model.unembed(h.float())`.
- `J-lens`: `model.unembed(h.float() @ J_layer.T)`.
- Thus both use the adapter's **full effective decoder**, including final
  LayerNorm/RMSNorm, affine head, and any logit softcap. There is no raw-linear
  shortcut and no model-family-specific handwritten alternative baseline.
- The final block is decoded **once per position chunk**, then reused for the
  reference model and both lenses. Its KL is exactly zero and agreement one.
  This follows `evaluate_paired`'s shared-final convention, even if a fitted
  final-layer matrix exists. The protocol assumes `unembed(final block output)`
  is the model's actual decoder.
- `model.encode(text, max_length=max_seq_len)` determines tokenization and
  truncation. Texts are not stripped, packed, concatenated, or windowed.
- Valid positions are those whose **input token** is not in tokenizer
  `all_special_ids`, supplemented by `bos_token_id`, `eos_token_id`, and
  `pad_token_id`. If these attributes are absent, no special IDs are inferred.
  The vocabulary distribution itself is **not filtered or renormalized**.
- Include the last non-special input position: these are next-token
  *distribution* comparisons, not scored next-token labels. No context skip
  or shifted target-token mask is applied.
- Special-only texts are counted but skipped without a forward pass.
  A corpus with no valid positions raises `ValueError`.
- Every valid token position has equal weight across the entire corpus. A
  100-token text contributes 100 times as much as a 1-token text. No averaging
  of per-text means, implicit dataset balancing, or dataset-level grouping.
  Call separately for distinct datasets; combine table means weighted by
  `n_tokens` if a pooled analysis is wanted.
- Nonfinite logits/log-probabilities/metrics raise `ValueError`, rather than
  yielding spurious top-1 results. All top-1 comparisons use **raw decoder
  logits**, before fp32 conversion or log-softmax, whose rounding can introduce
  ties. Actual raw-logit ties use the first vocabulary index.

### Exact `layers` table contract

Rows are ordered by lens (`"logit lens"`, then `"J-lens"`) and ascending layer.
There are `2 * L` rows for `L` selected blocks, including the final block.
Columns, in order:

| Column | Type | Meaning |
|---|---|---|
| `lens` | str | `"logit lens"` or `"J-lens"` |
| `layer` | int | Zero-based block index |
| `is_final` | bool | Shared final-model readout |
| `kl_model_to_lens` | float | Token mean of **KL(model distribution \|\| lens distribution)**, nats |
| `entropy` | float | Token mean of **lens** entropy, nats |
| `top1_agreement` | float | Fraction of positions matching the model argmax |
| `n_tokens` | int | Total valid positions contributing to this row |
| `n_texts` | int | Total input strings, including special-only strings |
| `n_texts_used` | int | Strings with at least one valid position |
| `weighting` | str | Always `"token"` |

All rows have the same counts. Natural logarithms are used. KL arithmetic is
fp32 and may exhibit tiny roundoff near zero; it is not artificially clipped.

### Exact `pairs` table contract

Return exactly **one logit-lens vs J-lens comparison per selected layer**,
`L` rows in ascending layer order. Every row has `lens_a="logit lens"`,
`lens_b="J-lens"`, and `layer_a == layer_b`. The final block is included.
There are no self-pairs or cross-layer comparisons and no cross-layer mode.
The function signature and column schema are unchanged. Pairs do **not** mean
independently forwarded texts or token pairs: the two readouts are compared at
the **same input position and layer**.

Columns, in order:

| Column | Type | Meaning |
|---|---|---|
| `lens_a` | str | First readout's lens name |
| `layer_a` | int | First readout's block index |
| `lens_b` | str | Second readout's lens name |
| `layer_b` | int | Second readout's block index |
| `symmetric_kl` | float | Token mean of **0.5 * (KL(a \|\| b) + KL(b \|\| a))**, nats |
| `top1_agreement` | float | Fraction of positions where the two argmaxes agree |
| `n_tokens` | int | Total valid positions |
| `n_texts` | int | Total input strings |
| `n_texts_used` | int | Strings with valid positions |
| `weighting` | str | Always `"token"` |

This is the **half-sum**, not the unscaled Jeffreys divergence and not
Jensen–Shannon divergence. The shared final-layer pair has exactly zero
symmetric KL and agreement one. This is a same-layer comparison table, **not**
a cross-layer similarity matrix. With `pairwise=False`, return an empty
DataFrame with these column names (empty column dtypes are not guaranteed);
the `layers` table is unaffected.

### Resource and precision behavior

- Inference mode only; no autograd graph or fitted-parameter updates.
- Layers are processed sequentially. Only the final reference and the current
  layer's two readouts are retained: distribution workspace is `O(C * V)`,
  **independent of layer count**, with `C=position_chunk_size`, `V=vocab_size`.
  Log-probabilities, probabilities, and temporary arithmetic arrays require
  several such tensors. No all-layer or corpus logits are retained.
- Position chunking bounds decoder/distribution workspace, **not vocabulary**:
  every `unembed` and softmax sees all `V` tokens for up to `C` positions.
  It also does not chunk the model forward: the complete truncated sequence
  runs once, and one text's selected block activations are held (`O(L*S*d)`,
  where `S` is encoded sequence length and `d=d_model`). Forward-internal
  attention/cache/workspace and model weights are additional costs.
- Requested Jacobians are cached on their activation devices for this call,
  adding `O(L * d_model**2)` fp32 storage (about 459 MiB for 47 GPT-2 XL
  matrices). The original lens tensors are not moved or mutated.
- Pair comparisons and scalar aggregate storage are **linear** in selected
  layer count. For `T` valid corpus positions, pair arithmetic is `O(L*T*V)`;
  decoding is roughly `O((2*L-1)*T*d*V)` for dense vocabulary heads, plus
  `O((L-1)*T*d**2)` transport and the model forwards. Each position chunk
  makes `2*L-1` decoder calls and at most `L` pair comparisons. Smaller chunks
  reduce memory but can hurt utilization and increase Python/kernel overhead;
  `pairwise=False` skips pair arithmetic, not the per-layer decoder calls.
- Adapter `unembed` owns model dtype/device conversion (as in existing
  evaluation); activations/Jacobians are promoted to fp32 for transport,
  distributions are fp32, and sums are fp64. Model weights/input activation
  storage dtype are not mutated; the adapter casts decoded residuals to its
  own compute dtype. Nonfinite residuals (including transport overflow),
  logits, log-probabilities, or aggregate metrics raise errors.
  Autocast is disabled on the forward/readout activation device.
  **Only CPU/CUDA compute devices are supported**: input IDs, captured readout
  activations, and decoded logits are checked explicitly. Other devices (including
  MPS, which lacks fp64 reductions) raise a descriptive `ValueError`, not an
  incidental allocation/reduction error. CPU/CUDA sharding/offloading remains
  subject to the adapter's own device support and correct `unembed` behavior.
- TF32 settings are **not changed**. For strict fp32 identity checks, temporarily
  use `torch.set_float32_matmul_precision("highest")`, then restore it. BF16
  model inference still uses BF16 model arithmetic. Geometry uses fp32 too.
- Progress is per text; geometry progress is per layer. `progress=False`
  disables both APIs' progress displays for tests/batch pipelines.

### GPT-2 XL rough scale (not measured timings)

For all 48 blocks, `d=1600`, `V=50,257`, and `C=8`:

- 96 layer rows but only **48 pair rows**, rather than 4,656 upper-triangle
  comparisons. No quadratic pair tensor is allocated.
- One fp32 `[C,V]` tensor is about **1.53 MiB**. The six persistent
  log-probability/probability arrays (reference plus two current readouts)
  total about **9.2 MiB**, plus several transient arrays and decoder workspace.
  This replaces about 147 MiB for just one old `[96,C,V]` array, of which the
  old implementation needed several. These estimates exclude allocator caches.
- Cached 47 Jacobians add about **459 MiB** fp32 device storage if copied from
  CPU; selected activations at `S=128` add about **37.5 MiB fp32** or
  **18.75 MiB BF16**. Model weights and forward workspace are separate.
- There are 95 decoder calls per chunk. Dense vocabulary projections alone
  cost about **1.96 trillion FLOPs per 128 valid positions**, counting a
  multiply-add as two FLOPs. The actual runtime depends strongly on GPU,
  precision, chunk size, adapter, and synchronization; this is not a timing
  prediction. The decoder/forward may dominate despite the pairwise saving.

`lens_vector_geometry`, in contrast, chunks **vocabulary rows**, not positions:
its `vocab_chunk_size=K` gives `O(K*d)` vector workspace, plus one `d*d`
Jacobian. It does not compute vocabulary distributions or softmax. Dense
geometry computation is roughly `O((L-1)*V*d**2)`.

## Linear-head geometry API

```python
lens_vector_geometry(
    model: LensModel,
    lens: JacobianLens,
    *,
    layers: Sequence[int] | None = None,
    unembedding_weight: Tensor | None = None,
    vocab_chunk_size: int = 4096,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> pandas.DataFrame
```

`W_U` has shape `[vocab_size, d_model]`. Corresponding token-vector rows are
compared between **`W_U` and `W_U @ J_layer`**, not `W_U @ J_layer.T`. This
orientation follows `(h @ J.T) @ W_U.T == h @ (W_U @ J).T` for a linear head.
The final block uses identity by convention.

**This statistic is explicitly linear-head-only geometry.** It excludes head
bias, normalization (including norm gain/centering), and softcap. In particular,
`W_U @ J` is **not** the effective linear map of `W_U * norm(Jh)`; that decoder
is nonlinear and has no single global token-vector representation. The
probabilistic API above includes these operations. Do not label this geometry
as the exact nonlinear decoded-lens direction. No Jacobian of normalization at
an unspecified activation, and no hidden folding of norm parameters, is used.

The protocol itself does not expose a linear head. Automatically use a plain
`torch.nn.Linear` at `model._lm_head` (the repository's HF/GPT-2 adapters) or
`model.lm_head`. Custom/quantized/nonlinear head classes are not assumed linear.
For another adapter, explicitly provide its true head's weight through
`unembedding_weight`; this is the caller's assertion about the linear part,
not permission to pass a nonlinear decoder as if it were a matrix. A missing
accessible head returns **`status="unsupported"` rows**, not guessed geometry.

The default compute device is the weight's device. **Compute must use CPU or
CUDA**, as cosine sums use fp64; other compute devices raise `ValueError` before
allocation/transfer. For weights stored on MPS, explicitly pass `device="cpu"`
to compute geometry on CPU. Only one fp32 layer matrix
and vocabulary chunks are transferred at a time; there is no full `W_U @ J`
allocation. Every vocabulary entry, including specials, has equal weight.
Pairs containing a zero-norm vector are excluded and counted. Malformed shapes
and nonfinite weights, Jacobians, transformed vectors, or norms raise errors.

### Exact geometry table contract

One row per selected layer (final always included), ascending layer order.
Columns, in order:

| Column | Type | Meaning |
|---|---|---|
| `layer` | int | Zero-based block index |
| `is_final` | bool | Identity convention used |
| `status` | str | `"ok"`, `"unsupported"`, or `"undefined"` |
| `reason` | str | Empty for ok; explanation otherwise |
| `mean_cosine` | float | Uniform mean over valid corresponding token vectors |
| `n_vocab` | int | Total vocabulary rows |
| `n_valid_vectors` | int | Rows with nonzero norms on both sides |
| `n_zero_vectors` | int | Rows excluded because either norm is zero |
| `weighting` | str | Always `"vocabulary"` |
| `convention` | str | Always `"linear_head_only: W_U vs W_U @ J"` |

For unsupported geometry, `mean_cosine=NaN` and all counts are **0 placeholders
for unavailable information**, not a measured empty vocabulary. For all-zero
geometry, `status="undefined"`, `mean_cosine=NaN`, and counts describe the
actual vocabulary. These explicitly labeled missing values are distinct from
nonfinite model outputs, which raise errors. A supported final-layer mean is
exactly 1 when at least one nonzero head row exists.
