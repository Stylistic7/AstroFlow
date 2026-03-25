import pytest
import datetime
from pathlib import Path
import pprint
from frozendict import frozendict

from dependency_graph import run_graph
from compute_graph import build_compute_graph
from context import context
from testing import copy_testdata_frames, make_synthetic_frames, print_tree
from customization.decisions import (
    DecisionType, DecisionOverride, BinaryChoice, ObjectType,
    get_decisions_for_object,
)
from customization.resolve import resolve_decisions
from customization.compute_graph import ALL_EXCLUSION_TYPES


@pytest.mark.heavy
def test_orion_pane(test_context):
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    make_synthetic_frames(
        prefix="dark",
        number=5,
        image_type="Dark",
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="bias",
        number=7,
        image_type="Bias",
        fill_value=50,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
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

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stack.
    graph = build_compute_graph()
    run_graph(graph, debug=True)

    assert len(context.stacks) == 1
    assert len(context.calibration_stacks) == 3
    assert len(context.targets) == 1

    filenames = list(
        str(f) for f in Path(context.output_dir).rglob("PostProcessed/*.xisf")
    )
    assert len(filenames) == 0

    stack_id = next(iter(context.stacks))
    context.shopping_cart = [("stack", stack_id)]

    print(f"### made stack {next(iter(context.stacks.values())).get()}")

    graph = build_compute_graph()
    run_graph(graph, debug=True)

    print_tree(context.output_dir)

    filenames = list(
        str(f) for f in Path(context.output_dir).rglob("PostProcessed/*.xisf")
    )
    assert len(filenames) == 1

    ui_stacks = context.ui_stacks
    assert len(ui_stacks) == 1
    assert list(ui_stacks.keys()) == [stack_id]

    wf = context["workflow_graphs"] or {}
    workflow_graph_ref = wf.get(("stack", stack_id))
    workflow_graph = workflow_graph_ref.get() if workflow_graph_ref is not None else None
    print(f"### stack workflow graph {workflow_graph}")


@pytest.mark.heavy
def test_resolve_decisions_filters_rotation_outlier(test_context):
    """resolve_decisions with a ROTATION_OUTLIER override excludes the flagged run."""
    # Two runs at different rotations — the 5-frame run should be flagged.
    make_synthetic_frames(
        ra=100, dec=50, prefix="normal", number=10, dither=0,
        rotation_angle_deg=90.0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="rotated", number=5, dither=0,
        rotation_angle_deg=97.0,
        start_datetime=datetime.datetime(2023, 12, 2, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stack = next(iter(context.stacks.values())).get()
    assert len(stack.all_frames()) == 15

    # Find the rotation outlier decision.
    stack_hash = next(iter(context.stacks))
    decisions = get_decisions_for_object(ObjectType.STACK, stack_hash)
    rotated_key = next(
        k for k, dp in decisions.items()
        if dp.decision_type == DecisionType.ROTATION_OUTLIER
        and dp.metadata["run_frame_count"] == 5
    )

    # Without override: all 15 frames survive (default is INCLUDE).
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, frozendict(), min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 15

    # With EXCLUDE override: only the 10 normal frames survive.
    overrides = frozendict({rotated_key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, overrides, min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 10


@pytest.mark.heavy
def test_resolve_decisions_filters_subframe_quality(test_context):
    """resolve_decisions applies quality filter, removing low-star-count frames."""
    # 10 good frames (1000 stars) + 5 bad frames (100 stars).
    # 75% threshold → cutoff at 750 stars → bad frames excluded.
    make_synthetic_frames(
        ra=100, dec=50, prefix="good", number=10, dither=0,
        rotation_angle_deg=90.0, num_stars=1000,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="bad", number=5, dither=0,
        rotation_angle_deg=90.0, num_stars=100,
        start_datetime=datetime.datetime(2023, 12, 1, 21, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stack = next(iter(context.stacks.values())).get()
    assert len(stack.all_frames()) == 15

    # Default: quality filter is active (INCLUDE means "apply filter").
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, frozendict(), min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 10

    # Override to EXCLUDE (skip filter): all 15 frames survive.
    stack_hash = next(iter(context.stacks))
    quality_key = (DecisionType.SUBFRAME_QUALITY, ObjectType.STACK, stack_hash, stack_hash)
    overrides = frozendict({quality_key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})
    resolved = resolve_decisions(stack, ALL_EXCLUSION_TYPES, overrides, min_frames_in_stack=1)
    assert len(resolved.all_frames()) == 15


@pytest.mark.heavy
def test_decisions_change_cache_key(test_context):
    """Changing decisions produces a different cache key for stack_files."""
    make_synthetic_frames(
        ra=100, dec=50, prefix="normal", number=10, dither=0,
        rotation_angle_deg=90.0,
        start_datetime=datetime.datetime(2023, 12, 1, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        ra=100, dec=50, prefix="rotated", number=5, dither=0,
        rotation_angle_deg=97.0,
        start_datetime=datetime.datetime(2023, 12, 2, 20, 0, 0),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stack = next(iter(context.stacks.values())).get()
    stack_hash = next(iter(context.stacks))

    # Find the rotation decision.
    decisions = get_decisions_for_object(ObjectType.STACK, stack_hash)
    rotated_key = next(
        k for k, dp in decisions.items()
        if dp.decision_type == DecisionType.ROTATION_OUTLIER
    )

    # Compute cache keys with different overrides — they must differ.
    from hashing import hash_call, hash_source_code
    from stacks.stacking import stack_files

    fn_hash = hash_source_code(stack_files)

    cal_stacks = context.calibration_stacks
    dev_settings = context.settings.development_settings
    min_frames = context.settings.clustering_settings.min_frames_in_stack

    key_no_override = hash_call(fn_hash, stack, cal_stacks, dev_settings, min_frames, frozendict())
    overrides = frozendict({rotated_key: DecisionOverride(choice=BinaryChoice.EXCLUDE)})
    key_with_override = hash_call(fn_hash, stack, cal_stacks, dev_settings, min_frames, overrides)

    assert key_no_override != key_with_override, (
        "Changing decisions must produce a different cache key"
    )
