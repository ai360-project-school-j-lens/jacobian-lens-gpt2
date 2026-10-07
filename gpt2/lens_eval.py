"""GPT-2 handwritten lenses, with shared architecture-independent evaluation.

Existing notebook imports remain available here; new multi-model notebooks
use ``jlens.evaluation`` and ``jlens.from_hf`` directly.
"""

from __future__ import annotations

import torch

from jlens.evaluation import (
    DATASETS as DATASETS,
    FIT_MIX as FIT_MIX,
    FIT_SOURCES as FIT_SOURCES,
    Lens as Lens,
    OPERATION_SYNONYMS as OPERATION_SYNONYMS,
    ROLE_NAMES as ROLE_NAMES,
    evaluate_all as evaluate_all,
    evaluate_paired as evaluate_paired,
    evaluate_readout as evaluate_readout,
    first_layer as first_layer,
    fit_with_progress as fit_with_progress,
    identity_lens as identity_lens,
    layer_hit_rate as layer_hit_rate,
    layer_mean as layer_mean,
    layer_median_rank as layer_median_rank,
    lens_slice as lens_slice,
    load_eval as load_eval,
    load_fit_prompts as load_fit_prompts,
    pass_at_k as pass_at_k,
    plot_layer_curves as plot_layer_curves,
    readout_position as readout_position,
    show_lens as show_lens,
    shown_layers as shown_layers,
    single_token_ids as single_token_ids,
    spelling_ids as spelling_ids,
    synonyms as synonyms,
    token_ranks as token_ranks,
    top_tokens_table as top_tokens_table,
)
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from .gpt2 import GPT2LensModel


@torch.no_grad()
def logit_lens(gpt: GPT2LensModel, text: str) -> torch.Tensor:
    """Handwritten logit lens: every block output through ln_f + unembedding, [n_layers, seq_len, vocab]."""
    res = gpt.forward(gpt.encode(text), output_hidden_states=True)
    # hidden_states[0] — эмбеддинги, их пропускаем; hidden_states[-1] у HF уже прошёл через ln_f,
    # поэтому последний слой — обычные логиты модели
    logits = [gpt.unembed(h[0]) for h in res.hidden_states[1:-1]]
    logits.append(gpt._lm_head(res.last_hidden_state[0]))
    return torch.stack(logits).float()


@torch.no_grad()
def jacobian_lens(gpt: GPT2LensModel, lens: JacobianLens, text: str) -> torch.Tensor:
    """Jacobian lens: block output h_l -> J_l @ h_l -> ln_f + unembedding, [n_layers, seq_len, vocab]."""
    # активации снимаются теми же хуками на gpt.layers, что и при fit,
    # поэтому J_l применяется ровно к тому, на чём он обучен
    with ActivationRecorder(gpt.layers, at=range(gpt.n_layers)) as recorder:
        gpt.forward(gpt.encode(text))
    logits = []
    for layer in range(gpt.n_layers):
        residual = recorder.activations[layer][0].float()
        # у последнего блока J = I — это логиты модели
        if layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits.append(gpt.unembed(residual))
    return torch.stack(logits).float()
