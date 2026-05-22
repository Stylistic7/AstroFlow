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
from products import makes_image
from products.identity import StackId, PresentationType

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID
from stacks.composite_stack import Stack

log = structlog.get_logger()

@node(auto_memory=True)
@makes_image(identity=StackId)
def post_process_stack_oss(stack: Stack, mock_pixinsight: bool) -> WithSidecar[FitsFilePath]:

    from stacks.stacking.oss import stack_files_oss
    master_path = stack_files_oss.filename_fn(stack)
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
@makes_image(identity=StackId)
def post_process_stack_oss_downscaled(
    stack: Stack, mock_pixinsight: bool
) -> WithSidecar[FitsFilePath]:
    """OSS gradient/background removal on the already-downscaled master,
    so a DownscaledMaster mosaic gets gradient correction cheaply."""

    # DOWNSCALED_MASTER comes from the OSS pipeline's downscale step;
    # for the current architecture (no OSS downscale op yet) read the
    # full-res master and let the post-process handle it.
    from stacks.stacking.oss import stack_files_oss
    master_path = stack_files_oss.filename_fn(stack)
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
