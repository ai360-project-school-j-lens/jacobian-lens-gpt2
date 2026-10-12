"""Cached, lazy-loading J-lens slices and a notebook prompt editor."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel
from jlens.strict_scoring import DecodedSpellings, ExplicitPromptModel
from jlens.vis import build_page, compute_slice, notebook_iframe


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


@dataclass
class RenderedSlice:
    """A self-contained page, its cache location, and rendering diagnostics."""

    page: str
    path: Path
    cache_hit: bool
    summary: dict[str, Any]
    unsupported_words: tuple[str, ...]


class SliceExplorer:
    """Render individual requested prompts, reusing weights and cached HTML.

    ``load_model`` returns a matching adapter and lens only on a cache miss.
    The manifest must describe model/tokenizer identity, lens hash, precision,
    fitting provenance, and implementation versions. It is copied on creation.
    ``validate_current`` can reject changed notebook configuration before any
    cache access. Call ``close`` before replacing a session to retire callbacks
    and release its model references.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        load_model: Callable[[], tuple[LensModel, JacobianLens]],
        manifest: Mapping[str, Any],
        cache_root: str | Path,
        vocab_size: int,
        bos_policy: str,
        max_seq_len: int = 512,
        title: str = "Interactive J-lens",
        description: str = "Full-vocabulary lexical ranks; final block uses the model readout.",
        validate_current: Callable[[], None] | None = None,
    ) -> None:
        if max_seq_len < 1 or vocab_size < 1:
            raise ValueError("max_seq_len and vocab_size must be positive")
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.title, self.description = title, description
        self._load_model = load_model
        self._validate_current = validate_current
        self._encoder = ExplicitPromptModel(
            SimpleNamespace(tokenizer=tokenizer), bos_policy=bos_policy,
        )
        self._runtime: tuple[ExplicitPromptModel, JacobianLens] | None = None
        self._spellings: DecodedSpellings | None = None
        self._closed = False
        self._widgets: list[Any] = []
        self._disconnect: list[Callable[[], None]] = []
        self.manifest = json.loads(_json({
            **manifest, "explorer_schema": 1, "bos_policy": bos_policy,
            "max_seq_len": max_seq_len, "vocab_size": vocab_size,
        }))
        self.cache_dir = Path(cache_root).expanduser() / _digest(_json(self.manifest))[:20]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _write(self.cache_dir / "manifest.json", _json(self.manifest) + "\n")

    def _require_current(self) -> None:
        if self._closed:
            raise RuntimeError("This explorer is closed; run the session and editor cells again.")
        if self._validate_current is not None:
            self._validate_current()

    def _model(self, prompt: str) -> tuple[ExplicitPromptModel, JacobianLens]:
        if self._runtime is None:
            adapter, candidate = self._load_model()
            if adapter.tokenizer is not self.tokenizer:
                raise ValueError("Loaded model and explorer must share the same tokenizer.")
            model = ExplicitPromptModel(adapter, bos_policy=self._encoder.bos_policy)
            required = set(range(model.n_layers - 1))
            if model.n_layers < 2 or candidate.d_model != model.d_model:
                raise ValueError("Need at least two model blocks and a matching lens width.")
            if required - set(candidate.source_layers):
                raise ValueError("Lens is missing inner layers for this model.")
            if set(candidate.source_layers) - set(range(model.n_layers)):
                raise ValueError("Lens contains out-of-range layers.")
            for layer, matrix in candidate.jacobians.items():
                if (matrix.shape != (model.d_model, model.d_model)
                        or not torch.isfinite(matrix).all()):
                    raise ValueError(f"Invalid/nonfinite Jacobian at layer {layer}.")
            # Move constants once. The final block always uses the model readout.
            device = model.encode(prompt, max_length=self.max_seq_len).device
            lens = JacobianLens(
                {layer: candidate.jacobians[layer].to(device) for layer in sorted(required)},
                n_prompts=candidate.n_prompts, d_model=candidate.d_model,
            )
            self._runtime = model, lens
        return self._runtime

    def render(
        self,
        prompt: str,
        *,
        pin_words: str = "",
        top_n: int = 8,
        max_tracked: int = 96,
        layer_stride: int = 1,
        last_n_tokens: int | None = 48,
        mask_display: bool = True,
        refresh: bool = False,
    ) -> RenderedSlice:
        """Compute or restore one exact prompt; pin one whole-token word per line.

        Whitespace is preserved and overlength prompts raise before loading
        weights. Missing whole-token spellings are reported, never reduced to
        their first subtoken. Tail windows limit the grid, not the model context.
        """
        self._require_current()
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Enter a nonempty prompt.")
        for name, value, minimum in [
            ("top_n", top_n, 1), ("max_tracked", max_tracked, 0),
            ("layer_stride", layer_stride, 1),
            ("last_n_tokens", last_n_tokens if last_n_tokens is not None else 1, 1),
        ]:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if top_n > self.vocab_size:
            raise ValueError("top_n exceeds the model vocabulary size.")
        if not isinstance(mask_display, bool):
            raise ValueError("mask_display must be boolean.")
        ids = self._encoder.encode_ids(prompt, max_length=self.max_seq_len)
        words = tuple(dict.fromkeys(w.strip() for w in pin_words.splitlines() if w.strip()))
        if words and self._spellings is None:
            self._spellings = DecodedSpellings(self.tokenizer, vocab_size=self.vocab_size)
        matches = {word: self._spellings(word) for word in words}
        pins = set().union(*matches.values())
        unsupported = tuple(word for word, token_ids in matches.items() if not token_ids)
        options = dict(
            top_n=top_n, max_tracked=max_tracked, layer_stride=layer_stride,
            last_n_tokens=last_n_tokens, mask_display=mask_display,
            max_seq_len=self.max_seq_len,
        )
        request = dict(
            prompt=prompt, options=options, pinned_token_ids=sorted(pins),
            title=self.title, description=self.description,
        )
        key = _digest(_json(request))
        path = self.cache_dir / f"slice-{key}.html"
        record_path = path.with_suffix(".json")
        if not refresh and path.is_file() and record_path.is_file():
            page = path.read_text(encoding="utf-8")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["request"] != request or record["html_sha256"] != _digest(page):
                raise ValueError("Slice cache failed verification; use refresh=True to recompute.")
            return RenderedSlice(page, path, True, record["summary"], unsupported)

        model, lens = self._model(prompt)
        with torch.inference_mode():
            data = compute_slice(model, lens, prompt, pinned_token_ids=pins, **options)
        page, _, _ = build_page(
            data, prompt, title=self.title, description=self.description, mode="embed",
        )
        summary = dict(
            prompt_tokens=len(ids), displayed_positions=data.seq_len,
            layers=data.layers, tracked_tokens=len(data.tracked_token_ids),
        )
        _write(path, page)
        _write(record_path, _json({
            "request": request, "summary": summary, "html_sha256": _digest(page),
        }) + "\n")
        return RenderedSlice(page, path, False, summary, unsupported)

    def display(self, prompt: str, *, height: int = 720, **options: Any) -> RenderedSlice:
        """Render a prompt and display its iframe plus cache/coverage details."""
        from IPython.display import display

        result = self.render(prompt, **options)
        summary = result.summary
        print(
            f"{'Cached' if result.cache_hit else 'Computed'}: "
            f"{summary['prompt_tokens']} context tokens; "
            f"{summary['displayed_positions']} displayed positions × "
            f"{len(summary['layers'])} layers."
        )
        if result.unsupported_words:
            print("No complete single-token spelling:", ", ".join(result.unsupported_words))
        print(f"Self-contained HTML: {result.path}")
        display(notebook_iframe(result.page, height=height))
        return result

    def widget(
        self, prompt: str, *, pin_words: str = "", height: int = 720,
        examples: Mapping[str, str] | None = None, **options: Any,
    ) -> Any:
        """Build an editor that runs only when Render is clicked; requires ipywidgets."""
        import ipywidgets as widgets
        from IPython.display import clear_output

        self._require_current()
        self._close_widgets()
        editor = widgets.Textarea(
            value=prompt, description="Prompt:",
            layout=widgets.Layout(width="100%", height="110px"),
        )
        pins = widgets.Textarea(
            value=pin_words, description="Track words:",
            placeholder="One whole-token word per line (optional)",
            layout=widgets.Layout(width="100%", height="65px"),
        )
        stride = widgets.BoundedIntText(
            value=options.get("layer_stride", 1), min=1, max=1000, description="Layer step:",
        )
        tail = widgets.BoundedIntText(
            value=options.get("last_n_tokens", 48) or 0,
            min=0, max=self.max_seq_len, description="Last tokens:",
        )
        mask = widgets.Checkbox(value=options.get("mask_display", True), description="Word-like tokens")
        button = widgets.Button(description="Render J-lens", button_style="primary", icon="play")
        output = widgets.Output()
        choices = widgets.Dropdown(options=[("Custom prompt", ""), *(examples or {}).items()],
                                   description="Example:")

        def choose(change):
            if change["new"]:
                editor.value = change["new"]

        def render(_):
            button.disabled = True
            try:
                with output:
                    clear_output(wait=False)
                    try:
                        selected = dict(options, layer_stride=stride.value,
                                        last_n_tokens=tail.value or None, mask_display=mask.value)
                        self.display(editor.value, pin_words=pins.value, height=height, **selected)
                    except Exception as error:
                        print(f"{type(error).__name__}: {error}")
            finally:
                button.disabled = self._closed

        choices.observe(choose, names="value")
        button.on_click(render)
        self._disconnect = [
            lambda: choices.unobserve(choose, names="value"),
            lambda: button.on_click(render, remove=True),
        ]
        controls = widgets.HBox([stride, tail, mask], layout=widgets.Layout(flex_flow="row wrap"))
        hint = widgets.HTML("Last tokens = 0 shows every position. Edit, then click Render; typing does not run inference.")
        ui = widgets.VBox([choices, editor, pins, controls, hint, button, output])
        self._widgets = [editor, pins, stride, tail, mask, button, output, choices, controls, hint, ui]
        return ui

    def _close_widgets(self) -> None:
        for disconnect in self._disconnect:
            disconnect()
        self._disconnect.clear()
        for widget in self._widgets:
            if hasattr(widget, "disabled"):
                widget.disabled = True
            widget.close()
        self._widgets.clear()

    def close(self) -> None:
        """Disable old controls and release the lazy-loaded model/lens."""
        self._closed = True
        self._close_widgets()
        self._runtime = None
