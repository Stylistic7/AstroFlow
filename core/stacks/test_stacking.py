import pytest
import datetime
from pathlib import Path
import pprint

from dependency_graph import run_graph
from compute_graph import build_compute_graph
from context import context
from testing import copy_testdata_frames, make_synthetic_frames, print_tree


@pytest.mark.exhaustive
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
        str(f) for f in Path(context.output_dir).rglob("PostProcessed/**/*.xisf")
    )
    assert len(filenames) == 0

    stack_id = next(iter(context.stacks))
    context.shopping_cart = [("stack", stack_id, "PixInsight Master")]

    print(f"### made stack {next(iter(context.stacks.values())).get()}")

    graph = build_compute_graph()
    run_graph(graph, debug=True)

    print_tree(context.output_dir)

    filenames = list(
        str(f) for f in Path(context.output_dir).rglob("PostProcessed/**/*.xisf")
    )
    assert len(filenames) == 1

    ui_stacks = context.ui_stacks
    assert len(ui_stacks) == 1
    assert list(ui_stacks.keys()) == [stack_id]

    wf = context["workflow_graphs"] or {}
    workflow_graph_ref = wf.get(("stack", stack_id))
    workflow_graph = workflow_graph_ref.get() if workflow_graph_ref is not None else None
    print(f"### stack workflow graph {workflow_graph}")
