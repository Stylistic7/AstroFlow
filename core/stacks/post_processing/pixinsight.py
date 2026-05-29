import os
from astropy.time import Time

from pix import Image
from pix.tasks import (
    ImageSolver,
    SpectrophotometricFluxCalibration,
    MultiscaleGradientCorrection,
    GradientCorrection,
)
from pix.tasks.filters import normalize_filter
from decorators import node
from decisions.settings import ProcessingConfig
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
    or downscaled stacker — the `image_scale` field encodes the
    actual arcsec/pixel of THIS image."""
    julian = float(Time(stack.oldest_date).jd)
    if master.plate_solution is not None:
        # `PlateSolution.image_scale` is the true pixel scale (with
        # rotation accounted for); using `wcs.pixel_scale_matrix[0,0]`
        # only reads the diagonal CD-matrix component, which collapses
        # to near-zero for rotated images and produces wildly wrong
        # hints.
        ra, dec = master.plate_solution.center
        arcsec_per_pixel = master.plate_solution.image_scale
    else:
        ra, dec = stack.center
        arcsec_per_pixel = stack.image_scale
    return dict(
        metadata_ra=float(ra), metadata_dec=float(dec),
        metadata_resolution=arcsec_per_pixel / 3600.0,
        metadata_observationTime=julian, metadata_epoch=julian,
    )


def _post_process_narrow(args, kwargs):
    """Cache-key hook: this op's only `config` dependency is the
    `mock_pixinsight` flag, so project `config` to that bool (in its
    arg slot). Keeps the key independent of unrelated config fields."""
    stack, config, master = args
    return (stack, config.development_settings.mock_pixinsight, master), kwargs


@node(resources={"PixInsight": 1}, auto_memory=True,
      cache_key_narrow=_post_process_narrow)
@makes_image(identity=StackId, display="Post-Processed")
def post_process_stack(
    stack: Stack, config: ProcessingConfig, master: ImageMetadata,
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

    Takes the whole `config` (uniform with the other heavyweight ops)
    and narrows the cache key to just `mock_pixinsight`. The filename
    is the call_hash over the narrowed bound args + `master`; two
    pipelines with the same `stack` + mock but different upstream
    masters produce different files (the masters' content_hashes
    differ).
    """
    mock_pixinsight = config.development_settings.mock_pixinsight
    output_filename = post_process_stack.filename_fn(
        stack, config, master,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    image = Image.open(master.filename, filter=stack.filter)
    solved = ImageSolver(image, **_solver_hints_from(master, stack))
    if normalize_filter(stack.filter) == "S":
        # MultiscaleGradientCorrection has no MARS reference data for
        # SII, so it fails deterministically on S-filter masters. Use
        # the classic GradientCorrection (no external data) instead.
        result = GradientCorrection(solved)
    else:
        result = MultiscaleGradientCorrection(
            SpectrophotometricFluxCalibration(solved)
        )
    result.save_to(output_filename, mock=mock_pixinsight)

    return WithSidecar(XisfFilePath(output_filename))
