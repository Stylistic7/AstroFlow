import ray
import numpy as np
from pathlib import Path

from context import context
from file_io.types import TiffFilePath
from file_io.tiff_file_io import write_tiff_image
from stacks.composite_stack import CompositeStack
from decorators import cached, measure_performance, create_thumbnail

from products.registry import makes_image, clear_registry, MediaType

clear_registry(MediaType.IMAGE)


def stack_filename(stack: CompositeStack) -> TiffFilePath:
    return TiffFilePath(f"master_{stack.name}.tiff")


@makes_image(CompositeStack).as_master(stack_filename)
@cached.remote
@measure_performance
@create_thumbnail
def stack_files(stack: CompositeStack) -> TiffFilePath:
    """Runs WBPP to stack raw images into a TIFF."""
    output_path = stack_files.filename_fn(stack)
    image = np.zeros((100, 100))
    write_tiff_image(output_path, image)

    return output_path


def test_single_stack(with_real_testdata) -> None:
    stack = next(iter(context.stacks.values())).get()

    filename = ray.get(stack_files.remote(stack))
    assert isinstance(filename, Path)

    thumbnail = stack_files.thumbnail_fn(stack)
    assert thumbnail is not None
    assert isinstance(thumbnail, bytes)
