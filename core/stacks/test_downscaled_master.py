"""Tests for the DOWNSCALED_MASTER quality level using real testdata."""
import pytest
import datetime
from pathlib import Path
from dataclasses import replace

from dependency_graph import run_graph
from compute_graph import build_compute_graph
from context import context
from testing import copy_testdata_frames, make_synthetic_frames, print_tree
from decisions.helpers import set_flat
from previews.types import QualityLevel


@pytest.mark.exhaustive
def test_downscaled_stack(test_context):
    """Stack a single pane at DOWNSCALED_MASTER quality with real PixInsight."""
    copy_testdata_frames(
        target_name="Orion Pane 1",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
        add_noise=False,
    )
    make_synthetic_frames(
        prefix="dark", number=5, image_type="Dark", fill_value=100,
        instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
        temperature=-15, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="bias", number=7, image_type="Bias", fill_value=50,
        instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
        temperature=-15, dest_dir=context.input_dirs[0],
    )
    make_synthetic_frames(
        prefix="flat", number=11, image_type="Flat", fill_value=50,
        start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
        instrument="ZWO ASI6200MM Pro", width_pixels=9576, height_pixels=6388,
        temperature=-15, dest_dir=context.input_dirs[0],
    )

    # First pass: identification.
    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.stacks) == 1
    assert len(context.calibration_stacks) == 3

    # Assign flat and trigger DOWNSCALED_MASTER stacking.
    stack_id = next(iter(context.stacks))
    stack = context.stacks[stack_id].get()
    flat_stack_id = next(
        sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"
    )
    for run_hash in stack.contents:
        set_flat(stack.stable_hash, run_hash, flat_stack_id)

    context.shopping_cart = [("stack", stack_id, "DownscaledMaster")]

    graph = build_compute_graph()
    run_graph(graph, debug=True)

    print_tree(context.output_dir)

    # Verify downscaled master was produced.
    downscaled_masters = list(Path(context.output_dir).rglob("DownscaledMaster/*.xisf"))
    assert len(downscaled_masters) >= 1, f"Expected downscaled master files, found: {downscaled_masters}"

    # Full-resolution masters should NOT exist (we only asked for downscaled).
    full_masters = list(Path(context.output_dir).rglob("Master/*.xisf"))
    assert len(full_masters) == 0, f"Unexpected full-resolution masters: {full_masters}"
