import pytest
import ray
import datetime

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from file_io.types import WithSidecar
from testing import make_synthetic_frames, copy_testdata_frames
from previews.stack_thumbnails import stack_thumbnail
from products import get_producer, MediaType, ensure_producers_registered
from previews.types import QualityLevel
from file_io.find_image_files import find_image_files

ensure_producers_registered()


@pytest.mark.exhaustive
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

    def check_thumbnail(stack):
        # stack_thumbnail's job now is just the cheap subframe SELECTION
        # — it returns the chosen frame. The PNG is rebuilt lazily by
        # the web server on first view (the actual lazy-thumbnail PNG
        # path is covered by test_endpoints'
        # test_lazy_thumbnails_regenerate_on_miss). Here we assert the
        # selection works and the SUBFRAME image producer is registered
        # so the lazy regen path can find it. The legacy
        # `.thumbnail_fn` attribute is gone (it was @create_thumbnail's;
        # producers now use @makes_image).
        file = ray.get(stack_thumbnail.remote(stack))
        assert isinstance(file, WithSidecar)
        assert get_producer(
            MediaType.IMAGE, type(stack), QualityLevel.SUBFRAME
        ) is not None

    for stack_ref in context.stacks.values():
        check_thumbnail(stack_ref.get())
