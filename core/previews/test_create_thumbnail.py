import pytest
import ray
import numpy as np
import datetime
import datetime

from compute_graph import build_compute_graph
from context import context
from dependency_graph import run_graph
from testing import copy_testdata_frames
from file_io.types import TiffFilePath
from file_io.tiff_file_io import write_tiff_image
from stacks.composite_stack import Stack
from decorators import cached, create_thumbnail


@cached.remote
@create_thumbnail
def stack_files(stack: Stack) -> TiffFilePath:
    output_path = TiffFilePath("foo.tiff").resolve_at(context.output_dir)
    image = np.zeros((100, 100))
    write_tiff_image(output_path, image)

    return output_path


@pytest.mark.exhaustive
def test_single_stack(test_context) -> None:
    copy_testdata_frames(
        prefix="light_run_0",
        target_name="NGC7635",
        start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
        dest_dir=context.input_dirs[0],
    )

    run_graph(build_compute_graph())

    stacks = context.stacks
    assert len(stacks) == 1
    stack = next(iter(stacks.values())).get()

    ray.get(stack_files.remote(stack))

    thumbnail = stack_files.thumbnail_fn(stack)
    assert thumbnail != None
    assert isinstance(thumbnail, bytes)
