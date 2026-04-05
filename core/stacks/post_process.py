import os
from astropy.time import Time

from pix import Image
from pix.tasks import (
    ImageSolver,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
)
from decorators import node, create_thumbnail
from stacks.composite_stack import Stack
from stacks.artifacts import stack_filename
from file_io.types import XisfFilePath, WithSidecar
from products import makes_image, get_filename_fn, MediaType
from previews.types import QualityLevel
import structlog

log = structlog.get_logger()


def _post_process_stack(master_path, stack: Stack, downscale_factor: int = 1) -> Image:
    image = Image.open(master_path)

    effective_scale = stack.image_scale * downscale_factor
    julian_date = int(Time(stack.oldest_date).jd)

    solver_kwargs = dict(
        metadata_ra=stack.center[0],
        metadata_dec=stack.center[1],
        metadata_resolution=effective_scale / 3600.0,
        metadata_observationTime=julian_date,
        metadata_epoch=julian_date,
    )

    if downscale_factor > 1:
        # Downscaled images have tiny stars — adjust solver for low resolution.
        solver_kwargs |= dict(
            solver_structureLayers=3,
            solver_sensitivity=0.9,
        )

    solved = ImageSolver(image, **solver_kwargs)

    calibrated = SpectrophotometricFluxCalibration(solved)
    return MultiscaleGradientCorrection(calibrated)


@makes_image(Stack).as_post_processed(stack_filename)
@node(resources={"PixInsight": 1})
@create_thumbnail
def post_process_stack(stack: Stack, mock_pixinsight: bool) -> WithSidecar[XisfFilePath]:
    from products import ensure_producers_registered
    ensure_producers_registered()
    master_fn = get_filename_fn(MediaType.IMAGE, Stack, QualityLevel.MASTER)
    master = master_fn(stack)

    output_filename = post_process_stack.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    result = _post_process_stack(master, stack, downscale_factor=1)
    result.save_to(output_filename, mock=mock_pixinsight)

    return WithSidecar(XisfFilePath(output_filename))
