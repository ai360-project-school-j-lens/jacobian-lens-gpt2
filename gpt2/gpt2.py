"""GPT-2 from the HuggingFace Hub as a :class:`jlens.protocol.LensModel`.

``GPT2LMHeadModel`` is laid out as::

    GPT2LMHeadModel
    ├── transformer: GPT2Model      # the decoder body ("encoder" of hidden states)
    │   ├── wte: Embedding          # token embedding
    │   ├── wpe: Embedding          # learned position embedding
    │   ├── h: ModuleList[GPT2Block]  # residual blocks -> LensModel.layers
    │   └── ln_f: LayerNorm         # final pre-unembed norm
    └── lm_head: Linear             # unembedding (weight tied to wte)

The lens hooks the outputs of ``transformer.h[i]`` (pre-``ln_f`` residual
stream) and decodes them with ``lm_head(ln_f(.))``.
"""

from __future__ import annotations

from typing import Any

import torch
import transformers
from torch import nn

from jlens.protocol import LensModel

MODEL_ID = "openai-community/gpt2"  # also: gpt2-medium, gpt2-large, gpt2-xl


class GPT2LensModel(LensModel):
    """Wraps a loaded ``GPT2LMHeadModel`` and its tokenizer.

    Holds references into ``hf_model``; nothing is copied. Puts the model in
    eval mode and freezes its parameters (the Jacobian fit only needs grads
    with respect to activations).
    """

    def __init__(self) -> None:
        pass

    def __init__(self, hf_model: transformers.GPT2LMHeadModel, tokenizer: Any) -> None:
        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        self._hf_model = hf_model
        self.tokenizer = tokenizer

        self.body: transformers.GPT2Model = hf_model.transformer
        self.layers: nn.ModuleList = self.body.h
        self._final_norm: nn.LayerNorm = self.body.ln_f
        self._embed_tokens: nn.Embedding = self.body.wte
        self._lm_head: nn.Linear = hf_model.lm_head

        self.n_layers: int = hf_model.config.n_layer
        self.d_model: int = hf_model.config.n_embd

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = MODEL_ID,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> GPT2LensModel:
        hf_model = transformers.GPT2LMHeadModel.from_pretrained(model_id, dtype=dtype)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        return cls(hf_model.to(device), tokenizer)

    def __repr__(self) -> str:
        return f"GPT2LensModel(n_layers={self.n_layers}, d_model={self.d_model})"

    @property
    def input_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        # GPT-2's tokenizer never prepends BOS; add <|endoftext|> by hand so the
        # first position acts as an attention sink. GPT-2's context is 1024.
        max_length = min(max_length, self._hf_model.config.n_positions)
        ids = self.tokenizer(text).input_ids[: max_length - 1]
        ids = [self.tokenizer.bos_token_id, *ids]
        return torch.tensor([ids], device=self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        # Runs wte + wpe -> h[0..n-1] -> ln_f; the lens reads block outputs via
        # hooks on self.layers, so the returned value is unused by jlens.
        return self.body(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        weight = self._lm_head.weight
        return self._lm_head(self._final_norm(residual.to(weight.device, weight.dtype)))
