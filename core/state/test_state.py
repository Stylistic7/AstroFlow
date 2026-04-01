import ray
import os
import datetime
from pathlib import Path
import time
from dataclasses import replace
from frozendict import frozendict

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from file_io.find_image_files import find_image_files
from customization.decisions import DecisionType, ObjectType, DecisionOverride


@ray.remote
def my_task(input):
    time.sleep(5)


def test_single_compute_graph(test_context):
    copy_testdata_frames(
        prefix="light_run_0",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
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

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 26

    graph = build_compute_graph()
    run_graph(graph)

    masters = list(Path(context.output_dir).rglob("*.xisf"))
    assert len(masters) == 0

    stacks = context.stacks
    assert len(stacks) == 1
    toplevel_stack = next(iter(stacks.values())).get()

    context.shopping_cart = [("stack", toplevel_stack.stable_hash)]

    # Flat selection is manual — override to assign the flat to every light run.
    flat_stack_id = next(
        (sid for sid, s in context.calibration_stacks.items() if s.image_type == "flat"),
        None,
    )
    overrides = {}
    for run_hash in toplevel_stack.contents:
        key = (DecisionType.CALIBRATION_FLAT, ObjectType.STACK, toplevel_stack.stable_hash, run_hash)
        overrides[key] = DecisionOverride(choice=flat_stack_id)
    context.settings = replace(context.settings, decision_overrides=frozendict(overrides))

    graph = build_compute_graph()
    run_graph(graph)

    masters = list(Path(context.output_dir).rglob("*.xisf"))
    for master in masters:
        print(f"master {master}")
    assert len(masters) == 5
