import os
from astropy.time import Time

from pix import Image
from pix.tasks import (
    ImageSolver,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
)
from decorators import node
from stacks.composite_stack import Stack
from file_io.types import XisfFilePath, WithSidecar
from products import makes_image
from products.identity import StackId, PresentationType
from previews.types import PIXINSIGHT, DOWNSCALE_FACTORS, DOWNSCALED_MASTER,MASTER,POST_PROCESSED,DOWNSCALED_POST_PROCESSED
import structlog

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID

log = structlog.get_logger()

def _post_process_stack(master_path, stack: Stack, downscale_factor: int = 1) -> Image:
    # Stack.filter is the authoritative grouping value; pix doesn't read
    # FILTER from disk (the caller is the only thing that knows it).
    image = Image.open(master_path, filter=stack.filter)

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

@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=StackId, display="Post-Processed")
def post_process_stack(stack: Stack, mock_pixinsight: bool) -> WithSidecar[XisfFilePath]:
    # Read upstream master directly. Each op exports its filename_fn;
    # consumers import the producing op explicitly instead of going
    # through a (identity, level, impl) registry lookup.
    from stacks.stacking.pixinsight import stack_files
    master = stack_files.filename_fn(stack)

    output_filename = post_process_stack.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    result = _post_process_stack(master, stack, downscale_factor=1)
    result.save_to(output_filename, mock=mock_pixinsight)

    return WithSidecar(XisfFilePath(output_filename))

@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=StackId, display="Post-Processed (Downscaled)")
def post_process_stack_downscaled(
    stack: Stack, mock_pixinsight: bool
) -> WithSidecar[XisfFilePath]:
    """Same gradient correction as `post_process_stack`, but on the
    already-downscaled master so a DownscaledMaster mosaic gets
    gradient removal cheaply (on the small image). Not pixel-identical
    to the full-res PostProcessed — `_post_process_stack`'s low-res
    solver branch (downscale_factor > 1) handles the tiny stars."""
    # Downscaled master comes from `stack_files_downscaled` — WBPP
    # ran on the already-downscaled input frames, so the output is
    # already at the smaller resolution.
    from stacks.stacking.pixinsight import stack_files_downscaled
    master = stack_files_downscaled.filename_fn(stack)

    output_filename = post_process_stack_downscaled.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    result = _post_process_stack(
        master, stack,
        downscale_factor=DOWNSCALE_FACTORS[DOWNSCALED_MASTER],
    )
    result.save_to(output_filename, mock=mock_pixinsight)

    return WithSidecar(XisfFilePath(output_filename))
