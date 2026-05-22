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
def post_process_stack_oss(
    stack: Stack, mock_pixinsight: bool, master,
) -> WithSidecar[FitsFilePath]:
    """Full-resolution OSS post-processing. `master` is the upstream
    `stack_files_oss` output passed as a data dep — no filename
    lookup needed."""
    output_filename = post_process_stack_oss.filename_fn(stack, mock_pixinsight)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    log.info("OSS post-processing stack", stack=str(stack), master=str(master.filename))

    data = read_image(master.filename).astype(np.float32)

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
    stack: Stack, mock_pixinsight: bool, master,
) -> WithSidecar[FitsFilePath]:
    """OSS gradient/background removal on the already-downscaled
    master. `master` is the upstream `stack_files_oss_downscaled`
    output as a data dep."""
    output_filename = post_process_stack_oss_downscaled.filename_fn(
        stack, mock_pixinsight,
    )
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    log.info("OSS post-processing downscaled stack",
             stack=str(stack), master=str(master.filename))

    data = read_image(master.filename).astype(np.float32)

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
