import pytest
import datetime
from dataclasses import replace

from frozendict import frozendict
from compute_graph import build_compute_graph
from context import context
from decisions.decisions import (
    DecisionKey, DecisionType, DecisionOverride, BinaryChoice, ObjectType,
    decisions_for_stack,
)
from decisions.helpers import exclude
from decisions.resolve import resolve_decisions
from decisions.compute_graph import ALL_EXCLUSION_TYPES, _is_quarantined
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from stacks.composite_stack import Stack


def test_quarantine_run(test_context) -> None:
    """Quarantining a run via DecisionKey.quarantine(parent_stack_id=...)
    must produce a key that survives config_for's per-stack filtering and
    actually excludes the run during resolve_decisions."""
    copy_testdata_frames(
        prefix="light_run_0",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat_run_0",
        number=11,
        image_type="Flat",
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    copy_testdata_frames(
        prefix="light_run_1",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 7, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat_run_1",
        number=13,
        image_type="Flat",
        start_datetime=datetime.datetime(2024, 2, 7, 18, 29, 43),
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    assert len(context.targets) == 1

    stack = next(iter(context.stacks.values())).get()
    assert len(stack.contents) == 2
    runs = list(stack.contents.keys())

    # Build quarantine key the same way the web endpoint does:
    # parent_stack_id = stack hash, subject_id = run hash.
    key = DecisionKey.quarantine(
        ObjectType.STACK, runs[0], parent_stack_id=stack.stable_hash,
    )
    assert key.parent_stack_id == stack.stable_hash, "must be the stack hash for config_for filtering"
    assert key.subject_id == runs[0], "must be the run hash for _apply_run_exclusions"

    overrides = frozendict(exclude(key))

    # Simulate config_for's per-stack filtering via decisions_for_stack.
    filtered = decisions_for_stack(overrides, stack.stable_hash)
    assert len(filtered) == 1, "quarantine key must survive per-stack filtering"

    # resolve_decisions should filter to only the second run.
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, filtered, min_frames_in_stack=1)
    assert resolved is not None
    assert list(resolved.contents.keys()) == [runs[1]]


def test_quarantine_triggers_restack(test_context) -> None:
    """End-to-end: quarantine a run, re-run the graph, verify the master
    file on disk is rewritten with new content.

    This is the contract the polygon viewer relies on: setting an
    exclusion override must invalidate the stack_files cache and
    actually re-execute, producing a fresh master at the same path.
    Without this, the green (output) polygon would never reflect a
    quarantine because it's read from the master's WCS header on disk.

    Uses mock_pixinsight stacking (which copies the first input frame
    as the 'stacked' output) so the test runs in seconds and doesn't
    need PixInsight."""
    import hashlib
    import time

    # Two runs with different rotation angles — mirrors the real-world
    # failure mode where one capture session was rotated relative to
    # another and the user wants to quarantine the rotated run.
    make_synthetic_frames(
        ra=100, dec=50, prefix="run_a", number=5, dither=0,
        rotation_angle_deg=90.0, num_stars=1000,
        start_datetime=datetime.datetime(2024, 1, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="run_b", number=5, dither=0,
        rotation_angle_deg=97.0, num_stars=200,
        start_datetime=datetime.datetime(2024, 1, 2, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    # First run: identification only (no shopping cart yet → no stack_files).
    run_graph(build_compute_graph())

    assert len(context.stacks) == 1
    stack_hash = next(iter(context.stacks))
    stack = context.stacks[stack_hash].get()
    assert len(stack.contents) == 2

    # Now drive stack_files via the shopping cart and re-run the graph.
    context.shopping_cart = [("stack", stack_hash, "PixInsight Master")]
    run_graph(build_compute_graph())

    # Locate the master file produced by stack_files (mock-stacked).
    # Post-identity-refactor the routing is just `<op_name>/<call_hash>
    # .xisf` — the per-type prefix is gone. Glob the directory for the
    # only master that was produced — there's just one stack in this
    # test.
    from pathlib import Path
    master_dir = Path(context.output_dir) / "stack_files"
    masters = list(master_dir.glob("*.xisf"))
    assert len(masters) == 1, (
        f"Expected exactly one stack_files master, found: {masters}"
    )
    master_path = masters[0]
    assert master_path.exists(), f"No master at {master_path} after first run"

    def file_hash(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def sidecar_image_metadata_content_hash(p):
        """Read the cached image_metadata.content_hash from the sidecar.

        This is the value compute_image_metadata stamped when it last
        ran on this file. If it doesn't change between graph runs, the
        sidecar shortcut returned stale data — i.e., compute_image_metadata
        trusted a sidecar that no longer matched the file content.
        """
        from file_io.sidecar import get_section
        im = get_section(p, "image_metadata")
        return im.get("content_hash") if isinstance(im, dict) else None

    original_hash = file_hash(master_path)
    original_mtime = master_path.stat().st_mtime
    original_meta_hash = sidecar_image_metadata_content_hash(master_path)
    assert original_meta_hash is not None, (
        "Expected compute_image_metadata to have written image_metadata "
        "to the sidecar after the first stack run"
    )

    # Identify which run's frames were copied into the master, and
    # quarantine THAT run, so the second re-stack must pick a different
    # frame and the master content must change.
    runs_by_oldest = sorted(stack.contents.items(), key=lambda kv: kv[1].oldest_date)
    first_run_hash = runs_by_oldest[0][0]

    key = DecisionKey.quarantine(
        ObjectType.STACK, first_run_hash, parent_stack_id=stack.stable_hash,
    )
    new_overrides = frozendict(exclude(key))
    context.settings = replace(context.settings, decision_overrides=new_overrides)

    # Coarse filesystems (HFS+) only have 1s mtime resolution.
    time.sleep(1.05)
    run_graph(build_compute_graph())

    # Post B-7-19, filenames are call-hash-based: decision overrides
    # flow into the cache_key_narrow → into the filename. A re-stack
    # with new decisions writes a NEW file at a NEW call_hash; the
    # original master file is left untouched (orphaned for GC). The
    # invariant we lock down here is "a fresh master was produced
    # whose content differs from the original."
    masters_after = sorted(
        master_dir.glob("*.xisf"), key=lambda p: p.stat().st_mtime,
    )
    new_masters = [m for m in masters_after if m != master_path]
    assert len(new_masters) >= 1, (
        f"Expected a new master file after quarantine + re-stack, "
        f"found only: {[p.name for p in masters_after]}"
    )
    new_master_path = new_masters[-1]
    new_hash = file_hash(new_master_path)
    new_meta_hash = sidecar_image_metadata_content_hash(new_master_path)

    assert new_hash != original_hash, (
        f"New master {new_master_path.name} has identical content to "
        f"the original {master_path.name} — quarantine had no effect"
    )
    assert new_meta_hash is not None, (
        "Expected compute_image_metadata to populate image_metadata "
        "in the new master's sidecar after the second stack run"
    )
    assert new_meta_hash != original_meta_hash, (
        f"Sidecar image_metadata.content_hash on new master matches "
        f"the original's ({original_meta_hash}); compute_image_metadata "
        f"returned a stale cached value instead of deriving fresh."
    )


def test_quarantine_target(with_orion_mosaic):
    """Quarantining a target via decision override marks it as quarantined."""
    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    target_hash = next(iter(context.targets))
    overrides = frozendict(exclude(DecisionKey.quarantine(ObjectType.TARGET, target_hash)))

    assert _is_quarantined(overrides, ObjectType.TARGET, target_hash)

    # Other targets are not affected.
    other_hashes = [h for h in context.targets if h != target_hash]
    for h in other_hashes:
        assert not _is_quarantined(overrides, ObjectType.TARGET, h)


def test_quarantine_stack(with_orion_mosaic):
    """Quarantining a stack via decision override marks it as quarantined."""
    stack_hash = next(iter(context.stacks))
    overrides = frozendict(exclude(DecisionKey.quarantine(ObjectType.STACK, stack_hash)))

    assert _is_quarantined(overrides, ObjectType.STACK, stack_hash)

    # Other stacks are not affected.
    other_hashes = [h for h in context.stacks if h != stack_hash]
    for h in other_hashes:
        assert not _is_quarantined(overrides, ObjectType.STACK, h)
