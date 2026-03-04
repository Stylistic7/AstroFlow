import ray
import datetime

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from file_io.types import WithSidecar
from testing import make_synthetic_frames, copy_testdata_frames
from previews.stack_thumbnails import (
    stack_thumbnail,
    calibration_stack_thumbnail,
)
from products.registry import get_producer, MediaType
from previews.types import QualityLevel
from file_io.find_image_files import find_image_files


def test_stack_dependency_graph(test_context) -> None:
    # These runs of dark and bias frames will be used by both runs of light
    # frames.
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

    filenames = find_image_files(context.input_dirs[0])
    assert len(filenames) == 42

    context.shopping_cart = [("stack", "c39bb8ef8c9266e6")]

    graph = build_compute_graph()
    run_graph(graph)

    assert len(context.stacks) == 1
    assert len(context.calibration_stacks) == 4

    def check_thumbnail(stack, fn):
        file = ray.get(stack_thumbnail.remote(stack))
        assert isinstance(file, WithSidecar)

        thumbnail = stack_thumbnail.thumbnail_fn(stack)
        assert thumbnail is not None
        assert isinstance(thumbnail, bytes)

        for mode in [QualityLevel.SUBFRAME, QualityLevel.MASTER]:
            fn = get_producer(MediaType.IMAGE, type(stack), mode)
            assert fn is not None
            thumbnail = fn.thumbnail_fn(stack)
            assert type(thumbnail) is bytes

            fn = get_producer(MediaType.IMAGE, type(stack), mode)
            assert fn is not None
            thumbnail = fn.thumbnail_fn(stack)
            assert type(thumbnail) is bytes

    for stack in context.stacks.values():
        check_thumbnail(stack, stack_thumbnail)
