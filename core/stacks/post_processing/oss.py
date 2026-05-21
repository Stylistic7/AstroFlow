"""OSS stack post-processing: background correction + plate solving.

Replaces ImageSolver + SpectrophotometricFluxCalibration +
MultiscaleGradientCorrection. Flux calibration is deferred.
"""
import os
import numpy as np
import structlog

import astropy.io.fits as pyfits
from photutils.background import Background2D, MedianBackground

import plate_solving
from decorators import node
from file_io.generic import read_image
from file_io.types import FitsFilePath, WithSidecar
from previews.types import OSS, DOWNSCALED_MASTER,MASTER,POST_PROCESSED,DOWNSCALED_POST_PROCESSED
from products import makes_image
from products.identity import StackId, PresentationType
from products.registry import get_filename_fn, MediaType

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID
from stacks.composite_stack import Stack

log = structlog.get_logger()


@node(auto_memory=True)
@makes_image(OSS, [POST_PROCESSED],
             identity=StackId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=FitsFilePath)
def post_process_stack_oss(stack: Stack, mock_pixinsight: bool) -> WithSidecar[FitsFilePath]:
    from products import ensure_producers_registered
    ensure_producers_registered()

    master_path = get_filename_fn(MediaType.IMAGE, StackId, MASTER, OSS)(stack)
    output_filename = post_process_stack_oss.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    log.info("OSS post-processing stack", stack=str(stack), master=str(master_path))

    data = read_image(master_path).astype(np.float32)

    # Background/gradient removal.
    try:
        bkg = Background2D(data, box_size=(64, 64), filter_size=(3, 3),
                           bkg_estimator=MedianBackground())
        data = data - bkg.background
    except Exception as e:
        log.warning("Background correction failed, leaving uncorrected", error=str(e))

    header = pyfits.Header()
    header["IMAGETYP"] = "Post-processed Light"
    header["STBLHASH"] = stack.stable_hash
    header["FILTER"] = stack.filter
    header["PIPELINE"] = "AstroFlow-OSS"
    header["RA"] = stack.center[0]
    header["DEC"] = stack.center[1]
    pyfits.PrimaryHDU(data=data, header=header).writeto(str(output_filename), overwrite=True)

    # Plate solve via the existing AstroFlow module (handles caching, etc).
    plate_solving.solve_field(output_filename)

    log.info("OSS post-processing complete", output=str(output_filename))
    return WithSidecar(FitsFilePath(output_filename))


@node(auto_memory=True)
@makes_image(OSS, [DOWNSCALED_POST_PROCESSED],
             identity=StackId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=FitsFilePath)
def post_process_stack_oss_downscaled(
    stack: Stack, mock_pixinsight: bool
) -> WithSidecar[FitsFilePath]:
    """OSS gradient/background removal on the already-downscaled master,
    so a DownscaledMaster mosaic gets gradient correction cheaply."""
    from products import ensure_producers_registered
    ensure_producers_registered()

    master_path = get_filename_fn(
        MediaType.IMAGE, StackId, DOWNSCALED_MASTER, OSS
    )(stack)
    output_filename = post_process_stack_oss_downscaled.filename_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    log.info("OSS post-processing downscaled stack",
             stack=str(stack), master=str(master_path))

    data = read_image(master_path).astype(np.float32)

    try:
        bkg = Background2D(data, box_size=(64, 64), filter_size=(3, 3),
                           bkg_estimator=MedianBackground())
        data = data - bkg.background
    except Exception as e:
        log.warning("Background correction failed, leaving uncorrected", error=str(e))

    header = pyfits.Header()
    header["IMAGETYP"] = "Post-processed Light"
    header["STBLHASH"] = stack.stable_hash
    header["FILTER"] = stack.filter
    header["PIPELINE"] = "AstroFlow-OSS"
    header["RA"] = stack.center[0]
    header["DEC"] = stack.center[1]
    pyfits.PrimaryHDU(data=data, header=header).writeto(str(output_filename), overwrite=True)

    plate_solving.solve_field(output_filename)

    log.info("OSS downscaled post-processing complete", output=str(output_filename))
    return WithSidecar(FitsFilePath(output_filename))
