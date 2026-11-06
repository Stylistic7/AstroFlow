"""End-to-end tests of the sweep CLI's orphan classification.

The live set is the TRACE (2026-08-17): every path traced for every cart item
and the pipelines implicitly carted per mosaic, unioned with the version-log
pins. The runtime census is gone — it under-pinned, needed
a completeness guard, and that guard deadlocked when a run failed. A trace
over-pins, which is the safe direction for a deletion authority.

These tests stub `traced_paths_for`; the exhaustive test below drives the full
Orion mosaic so the live set comes from a real trace.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from context import local_context as context
from cache import orphan_sweep


@pytest.fixture
def temp_output_dir(tmp_path):
    """Set `context.output_dir` to a fresh tmp dir for the test."""
    original = context.output_dir
    context.output_dir = str(tmp_path)
    yield tmp_path
    context.output_dir = original


@pytest.fixture(autouse=True)
def _clean_cart():
    """Reset the cart around each test."""
    original_cart = context.shopping_cart
    context.shopping_cart = []
    yield
    context.shopping_cart = original_cart


@pytest.fixture
def fake_trace(monkeypatch):
    """Stub the trace. Caller mutates `paths["live"]` — every cart item traces
    to that list, so tests declare liveness directly."""
    import dependency_graph.trace_paths as tp
    paths = {"live": []}
    monkeypatch.setattr(
        tp, "traced_paths_for",
        lambda item, *a, **kw: [{"path": p} for p in paths["live"]],
    )
    monkeypatch.setattr(orphan_sweep, "_pinned_versions", lambda: set())
    return paths


def _make_file(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_walk_and_classify_returns_unreferenced_files_only(temp_output_dir):
    live_path = _make_file(temp_output_dir / "mosaic" / "live.xisf")
    orphan_path = _make_file(temp_output_dir / "mosaic" / "orphan.xisf")
    live = {str(live_path)}

    orphans, _, _ = orphan_sweep.walk_and_classify(temp_output_dir, live)
    paths = {p for p, _ in orphans}

    assert paths == {str(orphan_path)}


def test_live_set_includes_traced_paths(temp_output_dir, fake_trace):
    """A cart item's traced paths are the live set."""
    live_path = temp_output_dir / "stack" / "master.xisf"
    _make_file(live_path)
    context.shopping_cart = [("stack", "s1", "P")]
    fake_trace["live"] = [str(live_path)]

    live = orphan_sweep.live_set(context)

    assert str(live_path) in live


def test_live_set_raises_on_unresolvable_cart_item(temp_output_dir, monkeypatch):
    """An item whose trace fails must RAISE, never drop silently out of the
    live set — dropping it would condemn all of its outputs as orphans."""
    import dependency_graph.trace_paths as tp
    from dependency_graph.trace_paths import TraceResolutionError
    _make_file(temp_output_dir / "stack" / "stranded.xisf")
    context.shopping_cart = [("stack", "s1", "P")]

    def boom(item, *a, **kw):
        raise TraceResolutionError("unresolvable")
    monkeypatch.setattr(tp, "traced_paths_for", boom)

    with pytest.raises(TraceResolutionError):
        orphan_sweep.live_set(context)


def test_sweep_refuses_when_live_set_empty(temp_output_dir, fake_trace):
    """Defensive floor: an empty cart and archive yield an empty live set.
    sweep() must NOT treat every file under
    output_dir as an orphan — it returns skipped_empty_live and deletes
    nothing."""
    orphan = _make_file(temp_output_dir / "stack" / "would_be_wiped.xisf")

    report = orphan_sweep.sweep(context, delete=True)

    assert report["skipped_empty_live"] is True
    assert report["deleted"] == 0
    assert orphan.exists()  # nothing deleted


def test_sweep_deletes_orphans_when_trace_defines_live(
    temp_output_dir, fake_trace,
):
    """With a non-empty live set, delete=True unlinks orphans while sparing
    the traced file — the empty-live floor doesn't block cleanup."""
    live = _make_file(temp_output_dir / "stack" / "keep.xisf")
    orphan = _make_file(temp_output_dir / "stack" / "drop.xisf")
    context.shopping_cart = [("stack", "s1", "P")]
    fake_trace["live"] = [str(live)]

    report = orphan_sweep.sweep(context, delete=True)

    assert report["skipped_empty_live"] is False
    assert live.exists()        # traced → live, spared
    assert not orphan.exists()  # unreferenced → deleted
    assert report["deleted"] >= 1


def test_full_sweep_classifies_correctly(temp_output_dir, fake_trace):
    """End-to-end: the trace covers the cart item; a file it does not name is
    the orphan."""
    cart_path = _make_file(temp_output_dir / "a" / "landed.xisf")
    orphan_path = _make_file(temp_output_dir / "c" / "orphan.xisf")

    context.shopping_cart = [("stack", "s1", "P")]
    fake_trace["live"] = [str(cart_path)]

    live = orphan_sweep.live_set(context)
    orphans, _, _ = orphan_sweep.walk_and_classify(temp_output_dir, live)
    orphan_paths = {p for p, _ in orphans}

    assert orphan_paths == {str(orphan_path)}


def test_walk_and_classify_returns_sizes(temp_output_dir):
    p = _make_file(temp_output_dir / "x.bin", size=512)
    orphans, _, _ = orphan_sweep.walk_and_classify(temp_output_dir, set())
    assert orphans == [(str(p), 512)]


def test_walk_and_classify_with_no_output_dir_returns_empty():
    """If output_dir is None (pre-init), sweep is a no-op."""
    orphans, by_dir, _ = orphan_sweep.walk_and_classify(None, set())
    assert orphans == []
    assert by_dir == {}


def test_include_scopes_to_named_dirs(temp_output_dir, fake_trace):
    """--include restricts classification to the named op directories;
    files outside are neither live nor orphan, just ignored."""
    included = _make_file(temp_output_dir / "place_image" / "a.tiff")
    excluded = _make_file(temp_output_dir / "stack_files" / "b.xisf")

    orphans, by_dir, _ = orphan_sweep.walk_and_classify(
        temp_output_dir, set(), include={"place_image"},
    )
    paths = {p for p, _ in orphans}

    assert str(included) in paths
    assert str(excluded) not in paths
    assert "place_image" in by_dir
    assert "stack_files" not in by_dir


# ---------------------------------------------------------------------
# Exhaustive lifecycle: full Orion mosaic, trace == disk, cart→archive
# ---------------------------------------------------------------------

import datetime
from dataclasses import replace

from compute_graph.compute_graph import build_compute_graph
from decisions.helpers import set_flat
from decisions.settings import DevelopmentSettings
from dependency_graph import run_graph
from dependency_graph.trace_paths import paths_for_cart_item
from mosaics.orion_mosaic import add_orion_mosaic_images


def _override_flats_for_all_stacks():
    """Copy of the helper from `mosaics/test_mosaic.py` — assigns the
    flat cal stack to every light stack run so the master pipeline
    can resolve a flat. Duplicated rather than imported so this file
    doesn't take a structural dependency on a sibling test module."""
    flat_stack_id = next(
        (sid for sid, s in context.calibration_stacks.items()
         if s.image_type == "flat"),
        None,
    )
    assert flat_stack_id is not None, "No flat calibration stack found"
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        for run_hash in stack.contents:
            set_flat(stack.stable_hash, run_hash, flat_stack_id)


def _disable_preview_graph():
    """Suppress the auto-dispatched Subframe preview graph for the
    duration of this test. With preview off, every on-disk output is
    from the explicit cart's Master pipeline — making the trace ==
    disk comparison clean (no Subframe outputs to bookkeep)."""
    settings = context.settings
    dev = replace(settings.development_settings, skip_preview_graph=True)
    context.settings = replace(settings, development_settings=dev)


def _master_pipeline_op_dirs() -> tuple[str, ...]:
    """Directories the Master pipeline writes primary outputs into.
    Used to filter disk-scan results for the trace==disk comparison."""
    return (
        "stack_files",
        "stack_files",
        "post_process_stack",
        "merge_target_channel",
        "place_image",
        "merge_images",
        "post_process_mosaic_color_channel",
    )


def _gather_disk_primaries(output_dir: Path) -> set[str]:
    """All on-disk primary outputs (xisf + tiff), excluding sidecars
    and tile pyramids.

    Pyramids are `<stem>_tiles.tiff` siblings (previews.create_tile_pyramid
    .pyramid_path_for) — the rename made them self-identifying, so the old
    stem-matches-an-xisf heuristic is gone. It silently broke when primaries
    became TIFFs (prefer-OSS-formats): pyramids of tiff primaries matched
    nothing and were miscounted as primaries."""
    primaries: set[str] = set()
    for d in _master_pipeline_op_dirs():
        full_dir = output_dir / d
        if not full_dir.exists():
            continue
        for p in full_dir.iterdir():
            if p.name.endswith("_tiles.tiff"):
                continue  # display pyramid — co-representation, not a primary
            if p.suffix in (".xisf", ".tiff", ".fits"):
                primaries.add(str(p))
            # .json/.context sidecars are intentionally excluded.
    return primaries


@pytest.mark.exhaustive
@pytest.mark.timeout(1200)  # builds + runs the full Orion mosaic pipeline twice
def test_orion_mosaic_trace_matches_disk_and_archive_lifecycle(test_context) -> None:
    """End-to-end validation of the pure-Merkle tracing-based live set.

    Phase 1 — Build & run the Orion mosaic Master pipeline. Assert
    `paths_for_cart_item` predicts EXACTLY the set of primary files
    on disk (trace == disk). This is the core correctness check for
    the convention `content_hash := producer.call_hash`: if the
    trace's call_hashes diverge from the runtime's at any node, the
    set difference will be non-empty.

    Phase 2 — Take the item out of the cart and classify: the cart is the
    whole declaration, so every primary the pipeline wrote is an orphan now.

    Phase 1 is ALSO the standing regression test for hash stability: the
    trace and the runtime must agree on every call_hash. The 2026-08
    generation re-keys showed up exactly as a trace/disk divergence.
    """
    _disable_preview_graph()
    add_orion_mosaic_images()

    # First pass: identification + cal-stack discovery.
    graph = build_compute_graph()
    run_graph(graph)

    mosaic = next(iter(context.mosaics.values())).get()
    item = ("mosaic", mosaic.stable_hash, "PixInsight Master")

    # Second pass: actually run the Master pipeline.
    context.shopping_cart = [item]
    _override_flats_for_all_stacks()
    graph = build_compute_graph()
    run_graph(graph)

    # --- Phase 1: trace == disk ---------------------------------------
    traced = {str(p) for p in paths_for_cart_item(item)}
    on_disk = _gather_disk_primaries(Path(context.output_dir))

    # Ops inside a `delete_intermediates` scope are TRACED but deliberately
    # NOT on disk — the scope reclaims them once its terminal completes (the
    # placements are multi-GB and feed only the merge). So their absence is
    # correct, not a divergence. Reuse the sweep's own definition rather than
    # a second list that could drift from it.
    from cache.orphan_sweep import SCOPE_INTERIOR_OPS

    def _is_scope_interior(path: str) -> bool:
        return os.path.basename(os.path.dirname(path)) in SCOPE_INTERIOR_OPS

    only_traced = {p for p in traced - on_disk if not _is_scope_interior(p)}
    only_disk = on_disk - traced

    if only_traced or only_disk:
        # Bucket both sides by op-directory so it's obvious where the
        # divergence is: is it one specific op that's off, or does the
        # mismatch cascade from the top? Print compactly so the test
        # log stays readable.
        from collections import defaultdict
        def bucket(paths):
            b = defaultdict(list)
            for p in sorted(paths):
                rel = os.path.relpath(p, str(context.output_dir))
                b[os.path.dirname(rel)].append(os.path.basename(p))
            return b
        msg = ["DIVERGENCE between trace and disk:"]
        msg.append(f"  trace produced {len(traced)} paths, disk has {len(on_disk)}")
        for op_dir, names in sorted(bucket(only_traced).items()):
            msg.append(f"  TRACE only [{op_dir}] ({len(names)}): {names[:3]}")
        for op_dir, names in sorted(bucket(only_disk).items()):
            msg.append(f"  DISK  only [{op_dir}] ({len(names)}): {names[:3]}")
        for op_dir, names in sorted(bucket(traced & on_disk).items()):
            msg.append(f"  MATCH      [{op_dir}] ({len(names)})")
        pytest.fail("\n".join(msg))

    # --- Phase 2: out of the cart, every primary is an orphan ----------
    # Liveness is what the TRACE says should exist for the cart, so removing
    # an item retires its files IMMEDIATELY — no pass has to run first.
    context.shopping_cart = []

    live = orphan_sweep.live_set(context)
    orphans, _by_dir, _ = orphan_sweep.walk_and_classify(
        Path(context.output_dir), live,
    )
    orphan_paths = {p for p, _ in orphans}

    assert on_disk, "Test setup produced no primaries"
    still_live = on_disk - orphan_paths
    assert not still_live, (
        f"{len(still_live)} primaries of an item out of the cart are still live "
        f"(first 3): {sorted(still_live)[:3]}"
    )


def test_identification_products_are_never_walked(temp_output_dir):
    """Dust measurements and star flats are the identification graph's, not
    any pipeline trace's: untraced, unreferenced, and never orphans."""
    dust = _make_file(temp_output_dir / "measure_dust_run" / "a.fits")
    flat = _make_file(temp_output_dir / "build_star_flat" / "b.xisf")
    orphan = _make_file(temp_output_dir / "mosaic" / "orphan.xisf")

    orphans, by_dir, _ = orphan_sweep.walk_and_classify(temp_output_dir, set())

    assert {p for p, _ in orphans} == {str(orphan)}
    assert "measure_dust_run" not in by_dir and "build_star_flat" not in by_dir
    assert dust.exists() and flat.exists()
