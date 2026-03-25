import os
from astropy.time import Time

from pix import Image
from pix.tasks import (
    ImageSolver,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
)
from decorators import node, create_thumbnail
from customization.settings import DevelopmentSettings
from stacks.composite_stack import Stack
from stacks.stacking import stack_files
from stacks.artifacts import stack_filename
from file_io.types import XisfFilePath, WithSidecar
from products import makes_image
import structlog

log = structlog.get_logger()


def _post_process_stack(master_path, stack: Stack) -> Image:
    image = Image.open(master_path)

    julian_date = int(Time(stack.oldest_date).jd)
    solved = ImageSolver(
        image,
        metadata_ra=stack.center[0],
        metadata_dec=stack.center[1],
        metadata_resolution=stack.image_scale / 3600.0,
        metadata_observationTime=julian_date,
        metadata_epoch=julian_date,
    )

    calibrated = SpectrophotometricFluxCalibration(solved)
    return MultiscaleGradientCorrection(calibrated)


@makes_image(Stack).as_post_processed(stack_filename)
@node(resources={"PixInsight": 1})
@create_thumbnail
def post_process_stack(stack: Stack, development_settings: DevelopmentSettings) -> WithSidecar[XisfFilePath]:
    master = stack_files.filename_fn(stack)
    log.info("Post-processing stack", master=master)

    output_filename = post_process_stack.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    if development_settings.dangerously_accept_existing_files and os.path.exists(output_filename):
        log.info("Dangerously skipping post-processing because file already exists", output_filename=output_filename)
        return WithSidecar(output_filename)

    result = _post_process_stack(master, stack)
    result.save_to(output_filename, mock=development_settings.mock_pixinsight)

    return WithSidecar(XisfFilePath(output_filename))
