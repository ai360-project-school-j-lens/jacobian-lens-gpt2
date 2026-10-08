# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Causal interventions in J-lens coordinates: concept swaps and thought injection.

The readout direction of a vocabulary token in the residual stream is what both
experiments are built on. The J-lens logit of token ``t`` at layer ``l`` is

    lens(h)[t] = u_t . norm(J_l h)

where ``u_t`` is the unembedding row and ``norm`` is the model's final norm. Pulling
the norm's gain ``gamma`` through to the left gives a direction in residual space::

    LayerNorm:  v_t = J_l^T C(gamma * u_t)      # C subtracts the channel mean
    RMSNorm:    v_t = J_l^T  (gamma * u_t)      # no centering, no bias

That difference is not cosmetic. A centering term applied to an RMSNorm model (or
omitted on a LayerNorm one) tilts every direction and the interventions below quietly
stop working, so :func:`final_norm_centers` decides it per model rather than per
notebook. The logit-lens control is the same formula with ``J_l = I``.

Interventions operate on a *band* of layers, as in the paper -- "at every token
position across a band of intermediate layers".
"""

from __future__ import annotations

from collections.abc import Sequence
from weakref import WeakKeyDictionary

import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

Positions = slice | Sequence[int]

__all__ = [
    "ClampSwap",
    "InjectDirection",
    "clear_jacobian_cache",
    "final_norm_centers",
    "lens_direction",
    "record_residuals",
    "swap_basis",
]

# ``JacobianLens`` holds its matrices in fp32 on the CPU. A band intervention asks for
# one per layer per trial, and a sweep runs thousands of trials, so moving them on every
# call costs more PCIe traffic than the whole experiment costs compute (26 MB per
# d_model=2560 matrix). They are immutable once fitted, so the device copy is cached,
# keyed weakly on the lens so dropping it frees the GPU memory.
_JACOBIAN_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def _jacobian_on(lens: JacobianLens, layer: int, device: torch.device) -> torch.Tensor:
    """``J_l`` as fp32 on ``device``, cached per (lens, layer, device)."""
    per_lens = _JACOBIAN_CACHE.setdefault(lens, {})
    key = (layer, str(device))
    if key not in per_lens:
        per_lens[key] = lens.jacobians[layer].to(device=device, dtype=torch.float32)
    return per_lens[key]


def clear_jacobian_cache() -> None:
    """Drop the cached device copies of the lens matrices."""
    _JACOBIAN_CACHE.clear()


def _module(model: LensModel, *names: str) -> nn.Module:
    """First of ``names`` the model exposes.

    ``HFLensModel`` names these ``_final_norm`` / ``_lm_head``; a bare HF module or a
    hand-rolled :class:`~jlens.protocol.LensModel` names them ``norm`` / ``lm_head``.
    Trying both in order is the head-discovery convention of :mod:`jlens.metrics`.
    """
    for name in names:
        found = getattr(model, name, None)
        if found is not None:
            return found
    raise AttributeError(
        f"{type(model).__name__} exposes none of {', '.join(names)}"
    )


def _randn_like(reference: torch.Tensor, generator: torch.Generator | None):
    """Gaussian noise shaped like ``reference``, drawn from ``generator``.

    A ``torch.Generator`` is bound to one device and cannot seed a tensor on
    another, so the draw happens on the generator's device and is moved after. That
    keeps a CPU-seeded run reproducible whichever device the model is on.
    """
    device = reference.device if generator is None else generator.device
    noise = torch.randn(
        reference.shape, generator=generator, device=device, dtype=reference.dtype
    )
    return noise.to(reference.device)


def final_norm_centers(model: LensModel) -> bool:
    """Whether the model's final norm subtracts the channel mean.

    Probed rather than read off the class name, so an unfamiliar norm module is
    classified correctly: adding the same constant to every channel is a no-op for a
    centering norm (LayerNorm) and changes the output for a scale-only one (RMSNorm).

    Args:
        model: Any model exposing a final norm (``_final_norm`` or ``norm``) and
            ``d_model``.
    """
    norm = _module(model, "_final_norm", "norm")
    weight = next(norm.parameters())
    probe = torch.randn(1, model.d_model, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        plain = norm(probe).float()
        shifted = norm(probe + 1.0).float()
    scale = plain.abs().max().clamp(min=1e-6)
    return bool((plain - shifted).abs().max() <= 0.05 * scale)


def lens_direction(
    model: LensModel,
    lens: JacobianLens | None,
    token_id: int,
    layer: int,
    *,
    mode: str = "J",
    centers: bool | None = None,
) -> torch.Tensor:
    """The residual-stream direction that carries ``token_id``'s lens logit.

    Args:
        model: Model the lens was fitted on.
        lens: Fitted lens. Unused (and may be ``None``) when ``mode`` is ``"logit"``.
        token_id: Vocabulary id whose direction to return.
        layer: Source layer, which selects ``J_l``.
        mode: ``"J"`` for the Jacobian-lens direction, ``"logit"`` for the
            logit-lens control (``J_l = I``).
        centers: Override the final-norm centering decision. Defaults to
            :func:`final_norm_centers`, which costs one tiny forward pass; pass it
            explicitly in a loop.

    Returns:
        A ``[d_model]`` float32 tensor on the unembedding's device.

    Raises:
        ValueError: If ``mode`` is not ``"J"`` or ``"logit"``, or if ``mode="J"``
            without a lens.
    """
    if mode not in ("J", "logit"):
        raise ValueError(f"mode must be 'J' or 'logit', got {mode!r}")
    if mode == "J" and lens is None:
        raise ValueError("mode='J' needs a fitted lens")
    if centers is None:
        centers = final_norm_centers(model)

    gamma = _module(model, "_final_norm", "norm").weight.float()
    direction = gamma * _module(model, "_lm_head", "lm_head").weight[token_id].float()
    if centers:
        direction = direction - direction.mean()
    if mode == "logit":
        return direction
    # J_l^T applied to the direction; `direction @ J` is (J^T direction) for a vector.
    return direction @ _jacobian_on(lens, layer, direction.device)


def swap_basis(
    model: LensModel,
    lens: JacobianLens | None,
    source_id: int,
    target_id: int,
    layer: int,
    *,
    mode: str = "J",
    centers: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``V = [v_s v_t]`` and its pseudoinverse, the 2-D coordinate system of a swap.

    The lens coordinates of a residual are ``c = V^+ h``; writing a modified ``c``
    back changes only the component of ``h`` in ``span(V)``.

    Returns:
        ``(V, V_pinv)`` of shapes ``[d_model, 2]`` and ``[2, d_model]``.
    """
    if centers is None:
        centers = final_norm_centers(model)
    columns = [
        lens_direction(model, lens, token_id, layer, mode=mode, centers=centers)
        for token_id in (source_id, target_id)
    ]
    V = torch.stack(columns, dim=1)
    return V, torch.linalg.pinv(V)


@torch.no_grad()
def record_residuals(
    model: LensModel,
    input_ids: torch.Tensor,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Block outputs at ``layers`` from one clean forward pass, detached."""
    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        return {layer: recorder.activations[layer].detach() for layer in layers}


class _BandHook:
    """Registers one forward hook per band layer and removes them on exit."""

    def __init__(
        self, model: LensModel, layers: Sequence[int], positions: Positions
    ) -> None:
        self._model = model
        self._layers = list(layers)
        self._positions = positions
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _delta(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        """The update to add to the patched positions of ``h``; ``[n_pos, d_model]``."""
        raise NotImplementedError

    def _hook(self, layer: int):
        def fn(module: nn.Module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            # Nothing to patch: with a KV cache, every step after the first passes a
            # single new token, so a prompt-position slice selects no rows. Returning
            # the output untouched keeps the intervention a prompt-only edit instead of
            # reducing over an empty slice and writing NaN into the stream.
            if h.shape[1] == 0 or h[0, self._positions].shape[0] == 0:
                return output
            patched = h.clone()
            delta = self._delta(layer, h)
            patched[0, self._positions] += delta.to(h.dtype)
            return patched if torch.is_tensor(output) else (patched, *output[1:])

        return fn

    def __enter__(self):
        try:
            for layer in self._layers:
                handle = self._model.layers[layer].register_forward_hook(
                    self._hook(layer)
                )
                self._handles.append(handle)
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ClampSwap(_BandHook):
    """Clamp a pair of lens coordinates to their swapped clean values across a band.

    After each band block, the residual is set to ``h + V(c* - V^+ h)`` with
    ``c* = c_clean + alpha * (swap(c_clean) - c_clean)``. At ``alpha=1`` the two
    coordinates are exchanged and the component orthogonal to ``span(V)`` is untouched.

    Clamping to the *clean* coordinates is what makes a band work. Adding
    ``alpha * V(swap(c) - c)`` to the already-patched stream at each layer undoes the
    swap at every second layer, and overshoots compound for ``alpha > 1``.

    Args:
        model: The model to hook.
        lens: Fitted lens; may be ``None`` when ``mode="logit"``.
        layers: Band layer indices.
        source_id: Token whose coordinate is swapped out.
        target_id: Token whose coordinate is swapped in.
        alpha: Swap strength; 1.0 exchanges the coordinates, 0.0 is a no-op.
        mode: ``"J"``, ``"logit"``, or ``"random"`` -- a norm-matched random update,
            the control for "any perturbation of this size would do it".
        clean: Clean-pass block outputs, from :func:`record_residuals`.
        positions: Which token positions to patch. Defaults to all of them, which is
            right for a model with no BOS; models with an attention-sink BOS want
            ``slice(1, None)``.
        generator: RNG for ``mode="random"``.
    """

    def __init__(
        self,
        model: LensModel,
        lens: JacobianLens | None,
        layers: Sequence[int],
        source_id: int,
        target_id: int,
        *,
        alpha: float = 1.0,
        mode: str = "J",
        clean: dict[int, torch.Tensor],
        positions: Positions = slice(None),
        centers: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(model, layers, positions)
        if mode not in ("J", "logit", "random"):
            raise ValueError(f"mode must be 'J', 'logit' or 'random', got {mode!r}")
        self._alpha = alpha
        self._mode = mode
        self._clean = clean
        self._generator = generator
        # "random" takes its basis from J so the update's size matches the J arm.
        basis_mode = "logit" if mode == "logit" else "J"
        if centers is None:
            centers = final_norm_centers(model)
        self._bases = {
            layer: swap_basis(
                model, lens, source_id, target_id, layer,
                mode=basis_mode, centers=centers,
            )
            for layer in layers
        }

    def _delta(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        V, V_pinv = self._bases[layer]
        rows = h[0, self._positions].float()
        coords = rows @ V_pinv.T
        clean = self._clean[layer][0, self._positions].float() @ V_pinv.T
        wanted = clean + self._alpha * (clean[:, [1, 0]] - clean)
        update = (wanted - coords) @ V.T
        if self._mode == "random":
            noise = _randn_like(update, self._generator)
            scale = update.norm(dim=-1, keepdim=True)
            update = noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-12) * scale
        return update


class InjectDirection(_BandHook):
    """Add a token's lens direction to the residual stream across a band.

    The injected vector is the unit-normalized lens direction scaled by the layer's
    mean residual norm over the patched positions, times ``strength``
    (``data/experiments/README.md``, verbal-introspection). Scaling by the layer's own
    norm keeps one ``strength`` comparable across layers of very different scale;
    ``strength=0`` is the control trial and is an exact no-op.

    Args:
        model: The model to hook.
        lens: Fitted lens; may be ``None`` when ``mode="logit"``.
        layers: Band layer indices.
        token_id: Vocabulary id of the concept to inject.
        strength: Multiplier on the layer's mean residual norm.
        mode: ``"J"``, ``"logit"``, or ``"random"`` (a fixed random direction,
            norm-matched to the J arm).
        positions: Token positions to inject at -- the user turn, for the
            introspection protocol.
        generator: RNG for ``mode="random"``.
    """

    def __init__(
        self,
        model: LensModel,
        lens: JacobianLens | None,
        layers: Sequence[int],
        token_id: int,
        *,
        strength: float = 1.0,
        mode: str = "J",
        positions: Positions = slice(None),
        centers: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(model, layers, positions)
        if mode not in ("J", "logit", "random"):
            raise ValueError(f"mode must be 'J', 'logit' or 'random', got {mode!r}")
        self._strength = strength
        if centers is None:
            centers = final_norm_centers(model)
        basis_mode = "logit" if mode == "logit" else "J"
        self._directions: dict[int, torch.Tensor] = {}
        for layer in layers:
            vector = lens_direction(
                model, lens, token_id, layer, mode=basis_mode, centers=centers
            )
            if mode == "random":
                vector = _randn_like(vector, generator)
            self._directions[layer] = vector / vector.norm().clamp(min=1e-12)

    def _delta(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        rows = h[0, self._positions].float()
        mean_norm = rows.norm(dim=-1).mean()
        direction = self._directions[layer].to(rows.device)
        return (self._strength * mean_norm) * direction.expand_as(rows)
