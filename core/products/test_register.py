import os
import ray
import numpy as np
from dependency_graph import run_graph

from decorators import cached
from products import makes_image, get_producer, MediaType
from compute_graph import build_compute_graph
from testing import copy_testdata_frames
from context import context
from file_io.types import TiffFilePath, XisfFilePath, ContentHashableFile
from file_io.tiff_file_io import write_tiff_image
from file_io.xisf_file_io import write_xisf_image
from stacks.composite_stack import CompositeStack
from previews.types import QualityLevel
from products.registry import clear_registry

clear_registry(MediaType.IMAGE)


def stack_filename_tiff(stack: CompositeStack) -> TiffFilePath:
    return TiffFilePath(f"stack_{stack.name}.tiff")


def stack_filename_xisf(stack: CompositeStack) -> XisfFilePath:
    return XisfFilePath(f"stack_{stack.name}.xisf")


@makes_image(CompositeStack).as_subframe(stack_filename_tiff)
@cached.remote
def stack_preview(stack: CompositeStack) -> TiffFilePath:
    """Loads the best individual FITS image from stack and saves as TIFF."""
    output_path = stack_preview.filename_fn(stack)
    image = np.zeros((100, 100))
    write_tiff_image(output_path, image)

    return output_path


@makes_image(CompositeStack).as_master(stack_filename_xisf)
@cached.remote
def stack_files(stack: CompositeStack) -> XisfFilePath:
    """Runs WBPP to stack raw images into an XISF."""
    output_path = stack_files.filename_fn(stack)
    image = np.zeros((100, 100, 1))
    write_xisf_image(output_path, image)

    return output_path


def test_single_stack(test_context) -> None:
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    run_graph(build_compute_graph())
    stack = next(iter(context.stacks.values()))

    filename = ray.get(stack_files.remote(stack))
    assert os.path.exists(filename)


def test_registry(test_context) -> None:
    copy_testdata_frames(target_name="NGC7635", dest_dir=context.input_dirs[0])
    run_graph(build_compute_graph())
    stack = next(iter(context.stacks.values()))

    fn = get_producer(MediaType.IMAGE, CompositeStack, QualityLevel.SUBFRAME)
    assert fn is not None

    filename = ray.get(fn.remote(stack))
    assert os.path.exists(filename)
    assert fn.is_cached(stack)
