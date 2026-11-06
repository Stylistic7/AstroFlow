"""Unit tests for `cache.orphan_sweep.sweep` — TRACE-as-authority (2026-08-17).

Liveness is what the TRACE says should exist, for every cart item and the
pipelines implicitly carted per mosaic. The runtime
census is gone: it recorded only what one complete pass landed, so it
under-pinned, needed a completeness guard, and that guard deadlocked (a failed
run stored no census, so the sweep freed nothing exactly when space was needed).
A trace over-pins instead — it names files not built yet — which is the safe
direction for a deletion authority.

Floors that keep a sweep from ever deleting live data:
  * an unresolvable cart item raises; the sweep aborts before deleting;
  * an EMPTY live set refuses to delete anything (misconfiguration signal);
  * version-log pins and each primary's siblings (sidecar + pyramid) stay live.

`traced_paths_for` and `_pinned_versions` are monkeypatched — tests never touch
Ray or the real durable cache.
"""
import types

import pytest

import cache.orphan_sweep as orphan_sweep
from dependency_graph.trace_paths import TraceResolutionError


def _ctx(tmp_path, cart=(), mosaics=None):
    out = tmp_path / "Processed"
    out.mkdir(exist_ok=True)
    return types.SimpleNamespace(
        shopping_cart=list(cart),
        mosaics=mosaics or {},
        output_dir=str(out),
    )


def _fake_trace(monkeypatch, by_item):
    """`{item: [paths]}`; an item absent from the map raises, exercising the
    strict-refusal floor."""
    import dependency_graph.trace_paths as tp

    def fake(item, *a, **kw):
        key = tuple(item)
        if key not in by_item:
            raise TraceResolutionError(f"cannot resolve {key!r}")
        return [{"path": p} for p in by_item[key]]

    monkeypatch.setattr(tp, "traced_paths_for", fake)


def _no_pins(monkeypatch):
    monkeypatch.setattr(orphan_sweep, "_pinned_versions", lambda: set())


def test_live_set_refuses_on_an_unresolvable_item(tmp_path, monkeypatch):
    """A cart item whose trace fails must RAISE, never silently drop out of
    the live set — dropping it would condemn all of its outputs as orphans."""
    _no_pins(monkeypatch)
    _fake_trace(monkeypatch, {})           # every item unresolvable
    ctx = _ctx(tmp_path, cart=[("target", "t1", "P")])
    with pytest.raises(TraceResolutionError):
        orphan_sweep.live_set(ctx)


def test_live_set_covers_the_cart_and_each_primary_s_siblings(tmp_path, monkeypatch):
    """The cart is the whole declaration: an item not in it pins nothing. Every
    primary drags its sidecar and pyramid in with it."""
    _no_pins(monkeypatch)
    _fake_trace(monkeypatch, {
        ("target", "t1", "P"): ["/o/a.tiff"],
        ("target", "t2", "P"): ["/o/b.tiff"],          # traceable, but not in the cart
    })
    ctx = _ctx(tmp_path, cart=[("target", "t1", "P")])

    live = orphan_sweep.live_set(ctx)

    assert {"/o/a.tiff", "/o/a.tiff.json", "/o/a_tiles.tiff"} <= live
    assert "/o/b.tiff" not in live


def test_mosaics_implicitly_carry_the_subframe_pipeline(tmp_path, monkeypatch):
    """`build_preview_graph` dispatches Subframe over EVERY mosaic — its own
    comment calls that "implicitly in the cart" — and those ops publish real
    files, so the live set must include them or the sweep eats the previews."""
    _no_pins(monkeypatch)
    _fake_trace(monkeypatch, {
        ("mosaic", "m1", "Subframe"): ["/o/sub.tiff"],
    })
    ctx = _ctx(tmp_path, mosaics={"m1": object()})

    assert "/o/sub.tiff" in orphan_sweep.live_set(ctx)


def test_sweep_deletes_only_untraced_files(tmp_path, monkeypatch):
    _no_pins(monkeypatch)
    out = tmp_path / "Processed" / "an_op"
    out.mkdir(parents=True)
    live_file = out / "keep.tiff"
    live_pyr = out / "keep_tiles.tiff"
    orphan = out / "dead.tiff"
    for f in (live_file, live_pyr, orphan):
        f.write_bytes(b"x" * 10)

    _fake_trace(monkeypatch, {("target", "t1", "P"): [str(live_file)]})
    ctx = _ctx(tmp_path, cart=[("target", "t1", "P")])

    report = orphan_sweep.sweep(ctx, delete=True)

    assert live_file.exists()
    assert live_pyr.exists()             # sibling of a traced primary
    assert not orphan.exists()
    assert report["deleted"] == 1
    assert report["footprints"] == {"target/t1/P": 20}      # the primary and its pyramid: what removal frees


def test_empty_live_set_refuses_to_delete(tmp_path, monkeypatch):
    """An empty cart is a misconfiguration signal, never license to wipe the
    output tree — every file would classify as an orphan."""
    _no_pins(monkeypatch)
    _fake_trace(monkeypatch, {})
    out = tmp_path / "Processed" / "an_op"
    out.mkdir(parents=True)
    victim = out / "everything.tiff"
    victim.write_bytes(b"precious")

    ctx = _ctx(tmp_path)                  # empty cart, archive, mosaics
    report = orphan_sweep.sweep(ctx, delete=True)

    assert report["skipped_empty_live"] is True
    assert report["deleted"] == 0
    assert victim.exists()


def test_version_log_pins_survive(tmp_path, monkeypatch):
    """The N-1 the UI is still showing must never be swept mid-rebuild."""
    out = tmp_path / "Processed" / "an_op"
    out.mkdir(parents=True)
    pinned = out / "last_good.tiff"
    pinned.write_bytes(b"still on screen")
    monkeypatch.setattr(orphan_sweep, "_pinned_versions",
                        lambda: {str(pinned)})
    _fake_trace(monkeypatch, {("target", "t1", "P"): ["/o/other.tiff"]})

    ctx = _ctx(tmp_path, cart=[("target", "t1", "P")])
    orphan_sweep.sweep(ctx, delete=True)

    assert pinned.exists()


def test_scope_interior_orphans_are_reported_as_a_gc_bug(tmp_path, monkeypatch):
    """A `delete_intermediates` subgraph cleans up after itself, so an interior
    surfacing here means that GC did not run. Report it rather than absorbing
    it silently (458 leaked `place_image` files, 2026-08-17)."""
    _no_pins(monkeypatch)
    interior = tmp_path / "Processed" / "place_image"
    interior.mkdir(parents=True)
    leaked = interior / "leaked.tiff"
    leaked.write_bytes(b"y" * 100)
    _fake_trace(monkeypatch, {("target", "t1", "P"): ["/o/a.tiff"]})

    ctx = _ctx(tmp_path, cart=[("target", "t1", "P")])
    report = orphan_sweep.sweep(ctx, delete=False)

    assert report["scope_gc_leaks"]["place_image"][0] == 1
    assert report["scope_gc_leaks"]["place_image"][1] == 100


def test_untraced_files_modified_since_the_boot_are_spared(tmp_path, monkeypatch):
    """A running node's scratch and half-written outputs are in no trace; they are newer than the
    app's boot. A crash leftover is older. The sweep deletes only the latter."""
    import os
    import time

    _no_pins(monkeypatch)
    out = tmp_path / "Processed" / "an_op"
    out.mkdir(parents=True)
    live_file, leftover, in_flight = out / "keep.tiff", out / "dead.tiff", out / ".siril-scratch.fit"
    for f in (live_file, leftover, in_flight):
        f.write_bytes(b"x" * 10)
    boot = time.time() - 3600.0
    os.utime(leftover, (boot - 7200.0, boot - 7200.0))                  # written before the boot
    _fake_trace(monkeypatch, {("target", "t1", "P"): [str(live_file)]})
    ctx = _ctx(tmp_path, cart=[("target", "t1", "P")])

    report = orphan_sweep.sweep(ctx, delete=True, spare_since=boot)

    assert live_file.exists() and in_flight.exists() and not leftover.exists()
    assert report["deleted"] == 1 and [p for p, _ in report["spared"]] == [str(in_flight)]
    assert report["by_dir"]["an_op"] == [1, 10, 1, 10]                   # the spared file is counted nowhere


def test_the_boot_is_read_from_the_ray_session_name():
    import datetime

    start = orphan_sweep.session_start("/tmp/ray/session_2026-11-05_16-15-14_181293_53689")
    assert start == datetime.datetime(2026, 11, 5, 16, 15, 14).timestamp()
    with pytest.raises(RuntimeError):
        orphan_sweep.session_start("/tmp/ray")
