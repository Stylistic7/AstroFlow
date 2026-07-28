"""End-to-end tests of the sweep CLI's orphan classification.

The live set is now derived by tracing each cart item's pipeline (see
`dependency_graph.trace_paths.paths_for_cart_item`) and unioning with
`context.archive`'s inline paths. These tests monkeypatch
`paths_for_cart_item` so we can exercise classification logic without
a real pipeline; a separate exhaustive test (`@pytest.mark.exhaustive`
below) drives the full Orion mosaic to validate the integration.
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
def _clean_cart_and_archive():
    """Reset cart and archive around each test."""
    original_cart = context.shopping_cart
    original_archive = context.archive
    context.shopping_cart = []
    context.archive = []
    yield
    context.shopping_cart = original_cart
    context.archive = original_archive


@pytest.fixture
def fake_trace(monkeypatch):
    """Stub `paths_for_cart_item` so cart items resolve to a fixed
    list. Caller mutates `bank[item] = [paths]`."""
    bank: dict[tuple, list[str]] = {}

    def fake(item, *, finals_only=False):
        return list(bank.get(tuple(item), []))

    monkeypatch.setattr(orphan_sweep, "paths_for_cart_item", fake)
    return bank


def _make_file(path: Path, size: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def test_walk_and_classify_returns_unreferenced_files_only(temp_output_dir):
    live_path = _make_file(temp_output_dir / "mosaic" / "live.xisf")
    orphan_path = _make_file(temp_output_dir / "mosaic" / "orphan.xisf")
    live = {str(live_path)}

    orphans, _ = orphan_sweep.walk_and_classify(temp_output_dir, live)
    paths = {p for p, _ in orphans}

    assert paths == {str(orphan_path)}


def test_live_set_includes_archive_paths_and_their_siblings(
    temp_output_dir,
):
    """Archive entries' paths are live. Their sibling sidecar
    (`.xisf.json`) and tile pyramid (`_tiles.tiff`) are also live so
    the sweep doesn't strand them when a primary is preserved."""
    primary = temp_output_dir / "mosaic" / "preserved.xisf"
    sidecar = temp_output_dir / "mosaic" / "preserved.xisf.json"
    # `_tiles.tiff` pyramid rename (previews.create_tile_pyramid.pyramid_path_for);
    # the old name was `preserved.tiff`.
    tile = temp_output_dir / "mosaic" / "preserved_tiles.tiff"
    _make_file(primary); _make_file(sidecar); _make_file(tile)

    context.archive = [("mosaic", "abc", "P", [str(primary)])]

    live = orphan_sweep.live_set(context)

    assert str(primary) in live
    assert str(sidecar) in live
    assert str(tile) in live


def test_live_set_traces_cart_items(temp_output_dir, fake_trace):
    """Cart items contribute their traced paths to the live set."""
    live_path = temp_output_dir / "stack" / "master.xisf"
    _make_file(live_path)
    item = ("stack", "xyz", "P")
    fake_trace[item] = [str(live_path)]
    context.shopping_cart = [item]

    live = orphan_sweep.live_set(context)

    assert str(live_path) in live


def test_cart_item_with_empty_trace_contributes_nothing(temp_output_dir):
    """An item in the cart whose trace returns [] (entity missing
    from context) contributes no paths."""
    orphan = _make_file(temp_output_dir / "stack" / "stranded.xisf")
    item = ("stack", "xyz", "P")
    # No monkeypatch → real paths_for_cart_item runs; returns []
    # because entity isn't in context.
    context.shopping_cart = [item]

    live = orphan_sweep.live_set(context)
    orphans, _ = orphan_sweep.walk_and_classify(temp_output_dir, live)

    assert {p for p, _ in orphans} == {str(orphan)}


def test_full_sweep_classifies_correctly(temp_output_dir, fake_trace):
    """End-to-end: cart's trace returns one master, archive preserves
    another, a third file is neither — only the third shows up as an
    orphan."""
    cart_path = _make_file(temp_output_dir / "a" / "cart.xisf")
    archive_path = _make_file(temp_output_dir / "b" / "archive.xisf")
    orphan_path = _make_file(temp_output_dir / "c" / "orphan.xisf")

    cart_item = ("stack", "1", "P")
    fake_trace[cart_item] = [str(cart_path)]
    context.shopping_cart = [cart_item]
    context.archive = [("mosaic", "2", "P", [str(archive_path)])]

    live = orphan_sweep.live_set(context)
    orphans, _ = orphan_sweep.walk_and_classify(temp_output_dir, live)
    orphan_paths = {p for p, _ in orphans}

    assert orphan_paths == {str(orphan_path)}


def test_walk_and_classify_returns_sizes(temp_output_dir):
    p = _make_file(temp_output_dir / "x.bin", size=512)
    orphans, _ = orphan_sweep.walk_and_classify(temp_output_dir, set())
    assert orphans == [(str(p), 512)]


def test_walk_and_classify_with_no_output_dir_returns_empty():
    """If output_dir is None (pre-init), sweep is a no-op."""
    orphans, by_dir = orphan_sweep.walk_and_classify(None, set())
    assert orphans == []
    assert by_dir == {}


def test_include_scopes_to_named_dirs(temp_output_dir, fake_trace):
    """--include restricts classification to the named op directories;
    files outside are neither live nor orphan, just ignored."""
    included = _make_file(temp_output_dir / "place_image" / "a.tiff")
    excluded = _make_file(temp_output_dir / "stack_files" / "b.xisf")

    orphans, by_dir = orphan_sweep.walk_and_classify(
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
from state import archive as archive_mod


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
    and tile-pyramid siblings of xisf primaries.

    Tile pyramids and primary tiffs share the `.tiff` extension; the
    disambiguation rule: a `.tiff` whose stem matches a `.xisf` in
    the same directory is a pyramid (the pyramid sits next to its
    primary). Standalone `.tiff` files (no co-located .xisf) are
    primaries (PlacedImage/place_image)."""
    primaries: set[str] = set()
    for d in _master_pipeline_op_dirs():
        full_dir = output_dir / d
        if not full_dir.exists():
            continue
        xisf_stems = {p.stem for p in full_dir.glob("*.xisf")}
        for p in full_dir.iterdir():
            if p.suffix == ".xisf":
                primaries.add(str(p))
            elif p.suffix == ".tiff" and p.stem not in xisf_stems:
                primaries.add(str(p))
            # .json sidecars and pyramid .tiffs (stem matches an
            # .xisf) are intentionally excluded.
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

    Phase 2 — Move the cart item to `archive` with `preserve="finals"`.
    Assert the archive entry contains `stack_files` masters and the
    final `post_process_mosaic_color_channel`, and nothing in between.

    Phase 3 — Run the sweep. Assert intermediates (post_process_stack,
    place_image, merge_images, merge_target_channel) are orphans;
    finals remain live.
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

    only_traced = traced - on_disk
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

    # --- Phase 2: cart → archive with preserve="finals" --------------
    archive_mod.apply_remove_from_cart(item, preserve="finals")

    assert list(context.shopping_cart or []) == []
    assert len(context.archive or []) == 1
    archived_entry = context.archive[0]
    assert tuple(archived_entry[:3]) == item
    archived_paths = set(archived_entry[3])

    # `finals_only=True` trace should equal what the archive captured.
    finals_traced = {
        str(p) for p in paths_for_cart_item(item, finals_only=True)
    }
    assert archived_paths == finals_traced

    # `stack_files` is now `final=True` — must be in the archive set.
    assert any("stack_files" in p for p in archived_paths)
    assert any(
        "post_process_mosaic_color_channel" in p
        for p in archived_paths
    )
    # Intermediates must NOT be in finals.
    for non_final_dir in (
        "post_process_stack",
        "place_image",
        "merge_images",
        "merge_target_channel",
    ):
        assert not any(non_final_dir in p for p in archived_paths), (
            f"{non_final_dir} should not be in finals"
        )

    # --- Phase 3: sweep classifies correctly --------------------------
    live = orphan_sweep.live_set(context)
    orphans, _by_dir = orphan_sweep.walk_and_classify(
        Path(context.output_dir), live,
    )
    orphan_paths = {p for p, _ in orphans}

    # Every archived (finals) path is live (covered by archive entry).
    for f in archived_paths:
        assert f in live, f"Archived path {f} missing from live set"
        assert f not in orphan_paths, (
            f"Archived path {f} unexpectedly orphan"
        )

    # Every Master intermediate primary is an orphan.
    intermediates_on_disk = {
        p for p in on_disk
        if any(d in p for d in (
            "post_process_stack",
            "place_image",
            "merge_images",
            "merge_target_channel",
        ))
    }
    assert intermediates_on_disk, "Test setup didn't produce intermediates"
    missing_from_orphans = intermediates_on_disk - orphan_paths
    assert not missing_from_orphans, (
        f"{len(missing_from_orphans)} intermediates not marked orphan "
        f"(first 3): {sorted(missing_from_orphans)[:3]}"
    )
