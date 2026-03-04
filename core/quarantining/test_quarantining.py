import datetime
import unittest

from hashing import hash_list
from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from stacks.composite_stack import CompositeStack
from mosaics.orion_mosaic import add_orion_mosaic_images


def test_quarantine_run(test_context) -> None:
    # The first run of light frames and its associated flats.
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

    # The second run of light frames and its associated flats.
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
    assert len(context.mosaics) == 0

    # The stack should have two runs.
    stack = next(iter(context.stacks.values()))
    assert len(stack.contents) == 2
    runs = list(stack.contents.keys())
    two_run_hash = hash_list([{stack.stable_hash: stack}], debug=True)

    # Quarantine the first run.
    context.quarantined_stacks = (runs[0],)

    run_graph(build_compute_graph())

    # The stack should have only the second run now.
    stack = next(iter(context.stacks.values()))
    assert list(stack.contents.keys()) == [runs[1]]
    one_run_hash = hash_list([{stack.stable_hash: stack}], debug=True)

    assert two_run_hash != one_run_hash


def test_quarantine_toplevel_target(test_context):
    add_orion_mosaic_images()
    run_graph(build_compute_graph())

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    assert all(len(s.dependencies) == 3 for s in stacks)

    context.quarantined_targets = (next(iter(context.targets)),)

    run_graph(build_compute_graph())

    assert len(context.targets) == 3
    assert len(context.mosaics) == 1

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    assert all(len(s.dependencies) == 3 for s in stacks)


def test_quarantine_calibration_stack(test_context):
    add_orion_mosaic_images()
    run_graph(build_compute_graph())

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    assert all(len(s.dependencies) == 3 for s in stacks)

    context.quarantined_calibration_stacks = [
        next(
            iter(
                s
                for s in context.calibration_stacks
                if context.calibration_stacks[s].image_type == "dark"
            )
        )
    ]
    run_graph(build_compute_graph())

    assert len(context.targets) == 4
    assert len(context.mosaics) == 1

    stacks = [s for s in context.stacks if type(s) is CompositeStack]
    assert all(len(s.dependencies) == 2 for s in stacks)
