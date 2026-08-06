"""Unit tests for `cache.orphan_sweep.sweep` — the global orphan
reconciliation (now completion/idle-only; the pre-run gate is gone).

Two properties keep an unattended sweep from ever deleting live data:
  * the live set includes the IMPLICIT preview items (`build_preview_graph`
    dispatches `Subframe` over every mosaic without a cart entry) — the
    regression that had the sweep deleting the whole subframe preview layer
    on every run, which the next run then deterministically rebuilt;
  * fail-loud floors: an unresolvable trace raises (never a silent partial
    live set), and an EMPTY resolved live set refuses to delete anything.

`paths_for_cart_item` and `_pinned_previous_versions` are monkeypatched —
tests must never trace real pipelines nor read the real durable cache."""
import types

import pytest

import cache.orphan_sweep as orphan_sweep
from dependency_graph.trace_paths import TraceResolutionError


def _ctx(tmp_path, cart=(), mosaics=()):
    out = tmp_path / "Processed"
    out.mkdir(exist_ok=True)
    return types.SimpleNamespace(
        shopping_cart=list(cart),
        archive=[],
        output_dir=str(out),
        mosaics={m: object() for m in mosaics},
    )


def _no_pins(monkeypatch):
    monkeypatch.setattr(orphan_sweep, "_pinned_previous_versions", lambda: set())


def test_implicit_subframe_outputs_survive_sweep(tmp_path, monkeypatch):
    """A Subframe store output — produced by the implicit per-mosaic preview
    pipeline, named by NO cart entry — must classify live and survive a
    delete sweep, while a genuinely unreferenced file in the same dir goes."""
    ctx = _ctx(tmp_path, cart=[("stack", "S1", "PixInsight Master")],
               mosaics=["M1"])
    out = tmp_path / "Processed"
    sub = out / "store_mosaic_color_channel_subframe"
    sub.mkdir()
    preview = sub / "abc123.tiff"
    preview.write_bytes(b"live")
    orphan = sub / "dead00.tiff"
    orphan.write_bytes(b"orphan")
    stack_master = out / "s1.xisf"
    stack_master.write_bytes(b"live")

    def fake_paths(item, strict=False, **kwargs):
        assert strict, "live set must always trace strict"
        if item == ("mosaic", "M1", "Subframe"):
            return [preview]
        if item == ("stack", "S1", "PixInsight Master"):
            return [stack_master]
        raise AssertionError(f"unexpected item {item!r}")

    monkeypatch.setattr(orphan_sweep, "paths_for_cart_item", fake_paths)
    _no_pins(monkeypatch)

    report = orphan_sweep.sweep(ctx, delete=True)

    assert preview.exists(), "implicit Subframe product was swept"
    assert stack_master.exists()
    assert not orphan.exists()
    assert report["deleted"] == 1


def test_implicit_items_enumerate_every_mosaic(tmp_path):
    ctx = _ctx(tmp_path, mosaics=["M1", "M2"])
    assert sorted(orphan_sweep._implicit_preview_items(ctx)) == [
        ("mosaic", "M1", "Subframe"),
        ("mosaic", "M2", "Subframe"),
    ]


def test_ctx_without_mosaics_yields_no_implicit_items(tmp_path):
    """A minimal ctx (CLI callers, tests) must not crash the live set."""
    ctx = types.SimpleNamespace(shopping_cart=[], archive=[], output_dir=None)
    assert orphan_sweep._implicit_preview_items(ctx) == []


def test_trace_failure_propagates_and_nothing_is_deleted(tmp_path, monkeypatch):
    """An unresolvable trace — here on an IMPLICIT preview item — must raise
    out of `sweep` before any walk/delete: a partial live set would classify
    that item's real outputs as orphans."""
    ctx = _ctx(tmp_path, mosaics=["M1"])
    victim = tmp_path / "Processed" / "keep.tiff"
    victim.write_bytes(b"x")

    def boom(item, strict=False, **kwargs):
        raise TraceResolutionError("entity absent from context")

    monkeypatch.setattr(orphan_sweep, "paths_for_cart_item", boom)
    _no_pins(monkeypatch)

    with pytest.raises(TraceResolutionError):
        orphan_sweep.sweep(ctx, delete=True)
    assert victim.exists()


def test_empty_live_set_refuses_to_delete(tmp_path, monkeypatch):
    """Empty cart + no mosaics resolves an empty live set: every file would
    classify as an orphan, so `sweep` must floor — zero deletions."""
    ctx = _ctx(tmp_path)
    victim = tmp_path / "Processed" / "keep.tiff"
    victim.write_bytes(b"x")

    monkeypatch.setattr(
        orphan_sweep, "paths_for_cart_item",
        lambda *a, **k: pytest.fail("nothing to trace"),
    )
    _no_pins(monkeypatch)

    report = orphan_sweep.sweep(ctx, delete=True)

    assert report["skipped_empty_live"] is True
    assert report["deleted"] == 0
    assert victim.exists()
