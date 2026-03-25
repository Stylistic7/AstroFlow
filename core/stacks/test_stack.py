import numpy as np
import pytest

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import make_synthetic_frames, copy_testdata_frames
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from stacks.types import StackHash, StackIdDependency


@pytest.mark.heavy
def test_one_dark_stack(test_context):
    make_synthetic_frames(
        prefix="dark",
        number=5,
        image_type="Dark",
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-15,
        exposure=300,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = context.calibration_stacks
    assert len(stacks) == 1
    stack = next(iter(stacks.values()))

    assert stack.content_hash != None
    assert len(stack.frames) == 5
    assert stack.image_type == "dark"
    assert stack.oldest_date != None
    assert stack.newest_date != None
    assert stack.camera == "ZWO ASI6200MM Pro"
    assert stack.mean_temperature == -15.0
    assert stack.exposure == (300,)
    assert stack.integration == 5 * 300
    assert stack.width_pixels == 9576
    assert stack.height_pixels == 6388


@pytest.mark.heavy
def test_one_light_one_dark(test_context):
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    make_synthetic_frames(
        prefix="dark",
        number=5,
        image_type="Dark",
        fill_value=100,
        instrument="ZWO ASI6200MM Pro",
        width_pixels=9576,
        height_pixels=6388,
        temperature=-10,
        exposure=300,
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    assert len(context.calibration_stacks) == 1

    toplevel_stack = next(
        ref.get() for ref in stacks.values() if isinstance(ref.get(), CompositeStack)
    )
    assert len(toplevel_stack.contents) == 1
    light_stack = next(iter(toplevel_stack.contents.values()))
    dark_stack = next(
        stack
        for stack in context.calibration_stacks.values()
        if isinstance(stack, SimpleStack) and stack.image_type == "dark"
    )

    assert toplevel_stack.image_type == "light"
    assert toplevel_stack.stable_hash != None
    assert toplevel_stack.content_hash != None
    assert len(toplevel_stack.contents) == 1
    assert toplevel_stack.oldest_date != None
    assert toplevel_stack.newest_date != None
    assert toplevel_stack.camera == "ZWO ASI6200MM Pro"
    assert toplevel_stack.filter == "H"
    assert toplevel_stack.mean_temperature == pytest.approx(-12.9)
    assert toplevel_stack.exposure == (300,)
    assert toplevel_stack.integration == 3 * 300
    assert toplevel_stack.width_pixels == 9576
    assert toplevel_stack.height_pixels == 6388
    assert toplevel_stack.name == "Bubble Nebula"
    assert np.allclose(toplevel_stack.center, [350.183329899, 61.2229195667])
    assert toplevel_stack.image_scale == pytest.approx(0.378, rel=1e-2)
    assert toplevel_stack.rotation_angle == pytest.approx(1.55, rel=1e-2)

    assert light_stack.image_type == "light"
    assert light_stack.content_hash != None
    assert len(light_stack.frames) == 3
    assert light_stack.dependencies == {"dark": StackIdDependency(stack_id=StackHash(dark_stack.stable_hash))}
    assert light_stack.oldest_date != None
    assert light_stack.newest_date != None
    assert light_stack.camera == "ZWO ASI6200MM Pro"
    assert light_stack.mean_temperature == pytest.approx(-12.9)
    assert light_stack.exposure == (300,)
    assert light_stack.integration == 3 * 300
    assert light_stack.width_pixels == 9576
    assert light_stack.height_pixels == 6388

    assert dark_stack.image_type == "dark"
    assert dark_stack.content_hash != None
    assert len(dark_stack.frames) == 5
    assert len(dark_stack.dependencies) == 0
    assert dark_stack.oldest_date != None
    assert dark_stack.newest_date != None
    assert dark_stack.camera == "ZWO ASI6200MM Pro"
    assert dark_stack.mean_temperature == -10.0
    assert dark_stack.exposure == (300,)
    assert dark_stack.integration == 5 * 300
    assert dark_stack.width_pixels == 9576
    assert dark_stack.height_pixels == 6388
