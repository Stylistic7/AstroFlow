import ray
from pathlib import Path
import os
from astropy.time import Time

from context import context
from pix.decorators import PixInsight
from pix.tasks import (
    BlurXTerminator,
    GradientCorrection,
    ImageSolver,
    SaveImages,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
)
from pix.classes import ImageList
from decorators import cached, measure_performance, create_thumbnail
from customization.types import CustomizedStack
from stacks.composite_stack import CompositeStack
from stacks.stacking import stack_files
from stacks.artifacts import stack_filename
from file_io.types import XisfFilePath
from file_io.generic import copy_image
from products import makes_image
from previews.types import QualityLevel


@PixInsight(slots=[1])
def _post_process_stack(
    images: ImageList,
    output_filenames: list[Path],
    stack: CustomizedStack,
) -> ImageList:
    # images = BlurXTerminator(images)
    # images = GradientCorrection(images)

    # TODO: Make this more self-contained inside the ImageSolver function.
    julian_date = int(Time(stack.oldest_date).jd)
    solved = ImageSolver(
        images,
        metadata_ra=stack.center[0],
        metadata_dec=stack.center[1],
        metadata_resolution=stack.image_scale / 3600.0,
        metadata_observationTime=julian_date,
        metadata_epoch=julian_date,
    )

    flux_calibrated = SpectrophotometricFluxCalibration(solved)
    gradient_corrected = MultiscaleGradientCorrection(flux_calibrated)

    SaveImages(gradient_corrected, output_filenames)


# TODO: Do post-processing immediately after each stack has been stacked.
@makes_image(CompositeStack).as_post_processed(stack_filename)
@cached.remote(resources={"PixInsight": 1})
@measure_performance
@create_thumbnail
def post_process_stack(stack: CustomizedStack) -> XisfFilePath:
    master = stack_files.filename_fn(stack)
    print(f"### post_process_stack, master {master}")

    output_filename = post_process_stack.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if context.dangerously_accept_existing_files and os.path.exists(
        output_filename
    ):
        print(
            f"### Dangerously skipping post-processing because "
            f"file {output_filename} already exists"
        )
        return output_filename

    if context.get("mock_pixinsight", False):
        copy_image(master, output_filename)
    else:
        _post_process_stack(
            [master],
            [output_filename],
            stack,
        )

    return XisfFilePath(output_filename)
