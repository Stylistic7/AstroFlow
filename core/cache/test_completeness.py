"""Unit tests for `cache.completeness` — the drain-time trace-vs-disk guard.

`traced_paths_for` is monkeypatched exactly as in `test_orphan_sweep` —
tests never touch Ray or the real durable cache.
"""
import types

import cache.completeness as completeness
from dependency_graph.trace_paths import TraceResolutionError


def _ctx(cart=(), mosaics=None):
    return types.SimpleNamespace(
        shopping_cart=list(cart), archive=[], mosaics=mosaics or {})


def _fake_trace(monkeypatch, by_item):
    """`{item: [entry dicts]}`; an absent item raises TraceResolutionError."""
    import dependency_graph.trace_paths as tp

    def fake(item, *a, **kw):
        key = tuple(item)
        if key not in by_item:
            raise TraceResolutionError(f"cannot resolve {key!r}")
        return by_item[key]

    monkeypatch.setattr(tp, "traced_paths_for", fake)


def _tap(path, op="op", display="Linear"):
    return {"path": path, "op_name": op, "display": display, "is_final": False}


def test_all_present_reports_nothing(tmp_path, monkeypatch):
    p = tmp_path / "a.tiff"
    p.write_bytes(b"x")
    item = ("mosaic", "m1", "WBPP")
    _fake_trace(monkeypatch, {item: [_tap(str(p))]})
    assert completeness.missing_products(_ctx(cart=[item])) == {}


def test_absent_tap_is_reported(tmp_path, monkeypatch):
    """The boot-7 shape: the trace promises a product the run never built."""
    item = ("mosaic", "m1", "WBPP")
    gone = str(tmp_path / "never_built.tiff")
    _fake_trace(monkeypatch, {item: [_tap(gone, op="blend_panels")]})
    missing = completeness.missing_products(_ctx(cart=[item]))
    assert list(missing) == [item]
    assert missing[item][0]["path"] == gone


def test_non_tap_intermediates_are_ignored(tmp_path, monkeypatch):
    """Reclaimed working values (`@node(ephemeral=True)` crops/feathers) are
    never taps — their absence is CORRECT, not incompleteness."""
    item = ("mosaic", "m1", "WBPP")
    entry = {"path": str(tmp_path / "reclaimed.tiff"), "op_name": "crop_panel",
             "display": None, "is_final": False}
    _fake_trace(monkeypatch, {item: [entry]})
    assert completeness.missing_products(_ctx(cart=[item])) == {}


def test_absent_final_is_reported_even_without_display(tmp_path, monkeypatch):
    item = ("mosaic", "m1", "WBPP")
    entry = {"path": str(tmp_path / "final.tiff"), "op_name": "finish",
             "display": None, "is_final": True}
    _fake_trace(monkeypatch, {item: [entry]})
    assert list(completeness.missing_products(_ctx(cart=[item]))) == [item]


def test_unresolvable_item_is_reported_not_raised(monkeypatch):
    """At drain, a trace that cannot resolve IS the finding (the finished run
    cannot have built what the item needs) — report it, never raise."""
    item = ("mosaic", "m1", "WBPP")
    _fake_trace(monkeypatch, {})
    missing = completeness.missing_products(_ctx(cart=[item]))
    assert "<unresolvable" in missing[item][0]["path"]


def test_log_completeness_never_raises(monkeypatch, capsys):
    def boom(ctx):
        raise RuntimeError("guard bug")
    monkeypatch.setattr(completeness, "missing_products", boom)
    completeness.log_completeness(_ctx(cart=[("mosaic", "m1", "WBPP")]))
    assert "guard failed to run" in capsys.readouterr().out