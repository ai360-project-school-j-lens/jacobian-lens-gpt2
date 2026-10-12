"""Offline prompt-editor, cache, and lazy model-loading regressions."""

import json
from unittest.mock import Mock

import pytest
import torch

import jlens.vis as vis
from jlens.evaluation import identity_lens
from jlens.interactive import SliceExplorer

from .tiny import TinyDecoder


class CharacterTokenizer:
    bos_token_id = 0
    all_special_ids = [0]

    def encode(self, text, *, add_special_tokens=False):
        return ([0] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) if i else "[BOS]" for i in ids)

    def get_vocab(self):
        return {chr(i): i for i in range(128)}


@pytest.fixture
def session(tmp_path, monkeypatch):
    # Model computation and HTML serialization remain real; no CDN request.
    monkeypatch.setitem(vis._TEMPLATE_FOR_MODE, "embed", vis._template("fetch"))
    model = TinyDecoder(n_layers=3, vocab_size=128)
    model.tokenizer = CharacterTokenizer()
    forward = Mock(wraps=model.forward)
    monkeypatch.setattr(model, "forward", forward)
    lens = identity_lens(model)
    loader = Mock(return_value=(model, lens))
    sessions = []

    def make(**options):
        settings = dict(
            tokenizer=model.tokenizer, load_model=loader,
            manifest={"model": "tiny", "revision": "seed-0"}, cache_root=tmp_path,
            vocab_size=128, bos_policy="prepend", max_seq_len=32,
        )
        settings.update(options)
        explorer = SliceExplorer(**settings)
        sessions.append(explorer)
        return explorer

    yield make, model, lens, loader, forward
    for explorer in sessions:
        explorer.close()


def bootstrap(page):
    return json.loads(page.split('id="bootstrap"')[1].split(">", 1)[1].split("</script>")[0])


def test_prompt_cache_skips_weights_and_forwards_across_sessions(session):
    make, model, _, loader, forward = session
    explorer = make()
    assert loader.call_count == 0
    first = explorer.render("ab ", pin_words="a\nnot-a-token", last_n_tokens=2)
    assert not first.cache_hit
    assert first.unsupported_words == ("not-a-token",)
    assert loader.call_count == forward.call_count == 1
    assert forward.call_args.args[0][0].tolist() == [0, ord("a"), ord("b"), ord(" ")]
    meta = bootstrap(first.page)["meta"]
    assert meta["prompt"] == "ab "
    assert meta["ctx_offset"] == 2
    assert meta["layers"][-1] == model.n_layers - 1
    assert {ord("a"), ord("A")} <= set(meta["pinned"])

    restored = make().render("ab ", pin_words="a\nnot-a-token", last_n_tokens=2)
    assert restored.cache_hit and restored.page == first.page
    assert loader.call_count == forward.call_count == 1
    explorer.render("ab", pin_words="a", last_n_tokens=2)
    explorer.render("ab ", pin_words="a", last_n_tokens=1)
    assert loader.call_count == 1 and forward.call_count == 3
    explorer.render("ab ", pin_words="a", last_n_tokens=2, refresh=True)
    assert loader.call_count == 1 and forward.call_count == 4


@pytest.mark.parametrize("bos,expected", [("none", 3), ("prepend", 4), ("tokenizer", 4)])
def test_special_tokens_and_whitespace_are_explicit(session, bos, expected):
    make, _, _, _, forward = session
    result = make(bos_policy=bos).render("a\n ")
    assert result.summary["prompt_tokens"] == expected
    assert forward.call_args.args[0][0, -2:].tolist() == [ord("\n"), ord(" ")]


@pytest.mark.parametrize("prompt,options,match", [
    ("", {}, "nonempty"), ("a" * 32, {}, "exceeding"),
    ("ab", {"top_n": 129}, "vocabulary"),
    ("ab", {"layer_stride": 0}, "layer_stride"),
    ("ab", {"last_n_tokens": 0}, "last_n_tokens"),
])
def test_invalid_input_does_not_load_weights(session, prompt, options, match):
    make, _, _, loader, forward = session
    with pytest.raises(ValueError, match=match):
        make().render(prompt, **options)
    loader.assert_not_called()
    forward.assert_not_called()


@pytest.mark.parametrize("change,match", [
    ("width", "width"), ("missing", "missing"), ("nonfinite", "nonfinite"),
])
def test_incompatible_lens_never_runs_a_model_pass(session, change, match):
    make, _, lens, _, forward = session
    if change == "width":
        lens.d_model += 1
    elif change == "missing":
        lens.source_layers = [1]
    else:
        lens.jacobians[0] = torch.full_like(lens.jacobians[0], float("nan"))
    with pytest.raises(ValueError, match=match):
        make().render("ab")
    forward.assert_not_called()


def test_stale_or_closed_session_rejects_even_cached_pages(session):
    make, _, _, loader, forward = session
    current = Mock()
    explorer = make(validate_current=current)
    explorer.render("ab")
    current.side_effect = RuntimeError("Configuration changed")
    with pytest.raises(RuntimeError, match="Configuration changed"):
        explorer.render("ab")
    explorer.close()
    assert explorer._runtime is None
    with pytest.raises(RuntimeError, match="closed"):
        explorer.render("ab")
    assert loader.call_count == forward.call_count == 1


def test_manifest_changes_and_corrupt_html_cannot_reuse_a_page(session):
    make, _, _, loader, forward = session
    explorer = make()
    first = explorer.render("ab")
    changed = make(manifest={"model": "tiny", "revision": "seed-1"}).render("ab")
    assert changed.path != first.path and not changed.cache_hit
    first.path.write_text("corrupt HTML")
    with pytest.raises(ValueError, match="cache failed verification"):
        explorer.render("ab")
    refreshed = explorer.render("ab", refresh=True)
    assert bootstrap(refreshed.page)["meta"] == bootstrap(first.page)["meta"]
    assert loader.call_count == 2 and forward.call_count == 3


def test_prompt_widget_runs_on_click_and_disconnects_old_controls(session, monkeypatch):
    widgets = pytest.importorskip("ipywidgets")
    make, _, _, loader, _ = session
    explorer = make()
    display = Mock()
    monkeypatch.setattr(explorer, "display", display)
    first = explorer.widget("ab", examples={"Example": "xy "})
    choices, editor, pins, controls, _, button, _ = first.children
    assert isinstance(editor, widgets.Textarea)
    editor.value = "changed\n "
    display.assert_not_called()
    choices.value = "xy "
    assert editor.value == "xy "
    pins.value = "x"
    controls.children[1].value = 0
    button.click()
    assert display.call_args.args == ("xy ",)
    assert display.call_args.kwargs["pin_words"] == "x"
    assert display.call_args.kwargs["last_n_tokens"] is None
    explorer.widget("another")
    button.click()
    assert button.disabled and display.call_count == 1
    explorer.close()
    loader.assert_not_called()
