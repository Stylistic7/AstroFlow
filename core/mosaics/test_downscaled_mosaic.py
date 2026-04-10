"""Integration test for DOWNSCALED_MASTER mosaic — full pipeline at reduced resolution."""
import pytest
from pathlib import Path
from frozendict import frozendict

from dependency_graph import run_graph
from compute_graph import build_compute_graph
from context import context
from mosaics.orion_mosaic import add_orion_mosaic_images
from decisions.helpers import set_flat, exclude
from decisions.decisions import DecisionKey, DecisionType, ObjectType, decisions_for_stack
from testing import print_tree


@pytest.mark.heavy
def test_downscaled_orion_mosaic(test_context):
    """Run the full mosaic pipeline at DOWNSCALED_MASTER quality."""
    add_orion_mosaic_images()

    # First pass: identification.
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    # Assign flats to all stacks.
    flat_stack_id = next(
        (sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"),
        None,
    )
    assert flat_stack_id is not None
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        for run_hash in stack.contents:
            set_flat(stack.stable_hash, run_hash, flat_stack_id)

    # Trigger DOWNSCALED_MASTER mosaic processing.
    context.shopping_cart = [("mosaic", mosaic.stable_hash, "DownscaledMaster")]

    graph = build_compute_graph()
    run_graph(graph, debug=True)

    print_tree(context.output_dir)

    output = Path(context.output_dir)

    # Calibration stacks should be in DownscaledMaster directory.
    cal_masters = list(output.rglob("CalibrationStack/DownscaledMaster/*.xisf"))
    assert len(cal_masters) >= 1, f"Expected calibration masters, found: {cal_masters}"

    # Light stack masters should be in DownscaledMaster directory.
    stack_masters = list(output.rglob("Stack/DownscaledMaster/*.xisf"))
    assert len(stack_masters) >= 1, f"Expected stack masters, found: {stack_masters}"

    # Full-resolution masters should NOT exist.
    full_masters = list(output.rglob("Stack/Master/*.xisf"))
    assert len(full_masters) == 0, f"Unexpected full-resolution masters: {full_masters}"

    # Note: placement and merging are not tested here because mock
    # stacking doesn't re-solve WCS at downscaled resolution, causing
    # projection failures. Real PixInsight handles this correctly.


@pytest.mark.heavy
def test_downscaled_orion_mosaic_produces_tiled_tiff(test_context):
    """The tiled TIFF that the web server serves must exist at the end
    of the DOWNSCALED_MASTER pipeline, alongside its XISF sibling."""
    add_orion_mosaic_images()

    # Pass 1: identification.
    run_graph(build_compute_graph())
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    # Assign flats.
    flat_stack_id = next(
        (sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"),
        None,
    )
    assert flat_stack_id is not None
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        for run_hash in stack.contents:
            set_flat(stack.stable_hash, run_hash, flat_stack_id)

    # Pass 2: full DOWNSCALED_MASTER mosaic pipeline.
    context.shopping_cart = [("mosaic", mosaic.stable_hash, "DownscaledMaster")]
    run_graph(build_compute_graph(), debug=True)

    print_tree(context.output_dir)

    output = Path(context.output_dir)
    xisfs = list(output.rglob("Mosaic/DownscaledMaster/*.xisf"))
    tiffs = list(output.rglob("Mosaic/DownscaledMaster/*.tiff"))

    # The per-filter XISF is the canonical final mosaic; the .tiff is
    # what get_image()/tile_metadata streams to the browser.
    assert len(xisfs) >= 1, f"Expected DownscaledMaster mosaic XISF(s), found: {xisfs}"
    assert len(tiffs) >= 1, f"Expected DownscaledMaster tiled TIFF(s), found: {tiffs}"
    # Every XISF should have a sibling tiled TIFF.
    for xisf in xisfs:
        sibling = xisf.with_suffix(".tiff")
        assert sibling.exists(), f"Missing tiled TIFF beside {xisf}"


@pytest.mark.heavy
def test_quarantine_run_only_invalidates_one_stack(test_context):
    """Quarantining a run in one stack must not cause other stacks in the
    same mosaic to be re-stacked. Uses dry_run to count how many nodes
    would actually execute."""
    add_orion_mosaic_images()

    # Pass 1: identification.
    run_graph(build_compute_graph())
    assert len(context.stacks) >= 4
    assert len(context.mosaics) == 1
    mosaic = next(iter(context.mosaics.values())).get()

    # Assign flats to all stacks.
    flat_stack_id = next(
        (sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"),
        None,
    )
    assert flat_stack_id is not None
    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        for run_hash in stack.contents:
            set_flat(stack.stable_hash, run_hash, flat_stack_id)

    # Pass 2: full DownscaledMaster mosaic — populates cache.
    context.shopping_cart = [("mosaic", mosaic.stable_hash, "DownscaledMaster")]
    run_graph(build_compute_graph())

    # Dry run with no changes — baseline: everything cached.
    _, baseline_stats = run_graph(build_compute_graph(), dry_run=True)
    baseline_nodes = baseline_stats.num_nodes_run

    # Pick the first stack and quarantine one of its runs.
    target_stack = next(iter(context.stacks.values())).get()
    assert len(target_stack.contents) >= 1
    quarantined_run = list(target_stack.contents.keys())[0]
    key = DecisionKey.quarantine(
        ObjectType.STACK, quarantined_run, parent_stack_id=target_stack.stable_hash,
    )
    from dataclasses import replace as dc_replace
    new_overrides = frozendict(
        dict(context.settings.decision_overrides) | exclude(key)
    )
    context.settings = dc_replace(context.settings, decision_overrides=new_overrides)

    # Dry run after quarantine — should invalidate only one stack's subtree.
    _, after_stats = run_graph(build_compute_graph(), dry_run=True)
    after_nodes = after_stats.num_nodes_run

    # The quarantined stack's subtree returns NeverRun and its descendants
    # are skipped. So after_nodes < baseline_nodes. But crucially, it should
    # NOT be dramatically smaller — only one stack's subtree is affected.
    # If decisions leaked to all stacks, after_nodes would be much smaller.
    skipped = baseline_nodes - after_nodes
    print(f"baseline={baseline_nodes} after_quarantine={after_nodes} skipped={skipped}")

    # Sanity: something was skipped (the quarantined stack's subtree).
    assert skipped > 0, "Expected at least some nodes to be skipped after quarantine"

    # Key assertion: most nodes still ran (cache hits). If all 4 stacks were
    # invalidated, skipped would be ~4x larger. We expect roughly 1/4 of
    # stacking nodes to be skipped (one stack out of four).
    assert after_nodes > baseline_nodes * 0.5, (
        f"Too many nodes skipped ({skipped} of {baseline_nodes}). "
        f"Decision change likely leaked to multiple stacks."
    )
