# Evaluations

Six original prompt distributions used to evaluate lens quality (§methods-comparison),
plus an easy multihop candidate set for smaller models. Each `{slug}.json` contains
prompts and annotations, not model activations or fitted artifacts.

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

## lens-eval-multihop-easy

[`lens-eval-multihop-easy.json`](lens-eval-multihop-easy.json), reproducibly built by
the executed [construction notebook](../../notebooks/model_agnostic/multihop_easy/multihop_easy_build.ipynb).

**Unfiltered candidates, not demonstrated small-model results.** There are 100
base items with two phrasings each: 32 geography, 20 animal, 20 opposite, 12
word-form, and 16 arithmetic items. Natural-language prompts request a final
property before describing the hidden bridge, avoiding the explicit
`company from → country` continuation slot in latent-bridge v2. Arithmetic items
ask for the product of an unevaluated sum or difference. Both the bridge and
answer labels are absent as whole words/numbers from the evaluation prompt.
These checks do not establish the absence of semantic shortcuts or predictable
bridge continuations at other positions.

Use the same intermediate-rank protocol as `multihop`: read **the final prompt
token** under both lenses, with the same layer range, and compare paired pass@k
curves/AUC and ranks. Final-answer correctness is an optional eligibility
diagnostic, not the lens-quality score. The set was not selected using either
lens. All 200 prompts have `name`, `prompt`, `target`, and `intermediates` fields;
each item annotates exactly one intermediate. `hop1_prompt` and `hop2_prompt` are
separate eligibility diagnostics, never evaluation context.

`base_id` groups paraphrases; average them before averaging base items. Report
per-family results and an equal-family macro average. `group_id` groups shared
bridge labels across phrasings and relations; use it for clustered uncertainty
and the supplied deterministic `dev`/`test` split. These splits are for choosing
evaluation settings, not lens fitting. Animal descriptions use familiar
prototypes, and some antonyms have alternative valid answers. Canonical labels
do not introduce new alias rules: in particular, this dataset name does not
activate the `order-ops` number-word expansion. Arithmetic intermediates use
number words so that they have whole-token spellings under both GPT-2 and
Qwen3-0.6B; their targets use digits. The executed notebook confirms coverage of
all 200 intermediate annotations for both tokenizers. Canonical targets have
coverage of 196/200 for GPT-2 and 168/200 for Qwen (multi-token numerals and/or
`South America` account for the gaps). Unsupported labels should remain
unscored, with coverage reported, rather than scored by a token prefix.

Load it with `load_eval(REPO_DIR, "multihop-easy")`. To evaluate it in the
model-agnostic dataset notebook, set
`evals = {"multihop-easy": load_eval(REPO_DIR, "multihop-easy")}` in its dataset
loading cell. The original default `DATASETS` list is unchanged. Existing
evaluators ignore the extra metadata; join grouping fields back by item name
for grouped reporting. The fixed item order ensures the legacy half-list-offset
control has a different bridge, but that control is not family-matched.

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
