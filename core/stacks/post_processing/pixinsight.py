import os
from astropy.time import Time

from pix import Image
from pix.tasks import (
    ImageSolver,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
)
from decorators import node
from metadata.classes import ImageMetadata
from stacks.composite_stack import Stack
from file_io.types import XisfFilePath, WithSidecar
from products import makes_image
from products.identity import StackId, PresentationType
import structlog

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID

log = structlog.get_logger()


def _solver_hints_from(master: ImageMetadata, stack: Stack) -> dict:
    """Plate-solver hints for `master`. Read from the master's
    already-computed plate solution (compute_image_metadata ran on
    this file as part of the @makes_image trace-hook chain), so the
    body doesn't need to know whether `master` came from a full-res
    or downscaled stacker — the WCS encodes the actual arcsec/pixel
    of THIS image."""
    julian = int(Time(stack.oldest_date).jd)
    if master.plate_solution is not None:
        from astropy.wcs import WCS
        wcs = WCS(master.plate_solution.header)
        cx, cy = master.width_pixels / 2.0, master.height_pixels / 2.0
        ra, dec = wcs.all_pix2world(cx, cy, 0)
        # Pixel scale of the master in arcsec/pixel.
        arcsec_per_pixel = abs(float(wcs.pixel_scale_matrix[0, 0])) * 3600.0
    else:
        ra, dec = stack.center
        arcsec_per_pixel = stack.image_scale
    return dict(
        metadata_ra=float(ra), metadata_dec=float(dec),
        metadata_resolution=arcsec_per_pixel / 3600.0,
        metadata_observationTime=julian, metadata_epoch=julian,
    )


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=StackId, display="Post-Processed")
def post_process_stack(
    stack: Stack, mock_pixinsight: bool, master: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """PixInsight post-processing of a stacked master.

    `master` arrives as an `ImageMetadata` (the @makes_image trace
    hook chains `compute_image_metadata` after every produced
    artifact, so by the time the master reaches this op at runtime
    it carries width_pixels, height_pixels, plate_solution, etc.).
    The body reads everything it needs from `master` and never
    needs to know HOW the master was produced — full-resolution
    stacking, downscaled stacking, or any future variant all look
    identical from here.

    Output filename is the call_hash over the FULL bound args
    including `master`. Two pipelines calling this op with the
    same `stack` + `mock_pixinsight` but different upstream masters
    produce DIFFERENT files: the masters' content_hashes differ,
    so the call_hash differs, so the filename differs.
    """
    output_filename = post_process_stack.filename_fn(
        stack, mock_pixinsight, master,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    image = Image.open(master.filename, filter=stack.filter)
    solved = ImageSolver(image, **_solver_hints_from(master, stack))
    result = MultiscaleGradientCorrection(
        SpectrophotometricFluxCalibration(solved)
    )
    result.save_to(output_filename, mock=mock_pixinsight)

    return WithSidecar(XisfFilePath(output_filename))
