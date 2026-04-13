"""OSS stack post-processing: plate solving + background correction.

Replaces PixInsight's ImageSolver + SpectrophotometricFluxCalibration +
MultiscaleGradientCorrection with solve-field (astrometry.net) and
photutils Background2D.
"""
import os
import subprocess
import tempfile
import numpy as np
import structlog
from pathlib import Path

import astropy.io.fits as pyfits
from astropy.wcs import WCS
from photutils.background import Background2D, MedianBackground

from decorators import node, create_thumbnail, create_tile_pyramid
from file_io.generic import read_image
from file_io.types import FitsFilePath, WithSidecar
from previews.types import QualityLevel, OSS
from products import makes_image
from products.registry import get_filename_fn, MediaType
from stacks.artifacts import stack_filename_fits
from stacks.composite_stack import Stack

log = structlog.get_logger()


def _solve_field(fits_path, ra_hint, dec_hint, scale_hint):
    """Run local astrometry.net solve-field for plate solving.

    Returns True if solve succeeded, False otherwise.
    The input file is modified in-place with WCS keywords added.
    """
    cmd = [
        "solve-field",
        str(fits_path),
        "--overwrite",
        "--no-plots",
        "--crpix-center",
        "--scale-units", "arcsecpix",
        "--scale-low", str(scale_hint * 0.8),
        "--scale-high", str(scale_hint * 1.2),
        "--ra", str(ra_hint),
        "--dec", str(dec_hint),
        "--radius", "5",
        "--timeout", "60",
        "--new-fits", str(fits_path),  # overwrite in place
        "--temp-dir", tempfile.gettempdir(),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and "solved" in result.stdout.lower():
            log.info("Plate solve succeeded", path=str(fits_path))
            return True
        else:
            log.warning("Plate solve failed", returncode=result.returncode,
                       stderr=result.stderr[:500])
            return False
    except FileNotFoundError:
        log.warning("solve-field not installed, skipping plate solve")
        return False
    except subprocess.TimeoutExpired:
        log.warning("Plate solve timed out")
        return False


def _background_correct(data):
    """Remove sky gradients using photutils Background2D."""
    try:
        bkg = Background2D(
            data, box_size=(64, 64),
            filter_size=(3, 3),
            bkg_estimator=MedianBackground(),
        )
        return data - bkg.background
    except Exception as e:
        log.warning("Background correction failed, returning uncorrected", error=str(e))
        return data


@node
@makes_image(Stack, OSS, [QualityLevel.POST_PROCESSED], stack_filename_fits)
@create_thumbnail
@create_tile_pyramid
def post_process_stack_oss(stack: Stack, mock_pixinsight: bool) -> WithSidecar[FitsFilePath]:
    from products import ensure_producers_registered
    ensure_producers_registered()

    master_fn = get_filename_fn(MediaType.IMAGE, Stack, QualityLevel.MASTER, OSS)
    master_path = master_fn(stack)

    output_fn = post_process_stack_oss.filename_fn
    output_filename = output_fn(stack)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    log.info("OSS post-processing stack", stack=str(stack), master=str(master_path))

    # Read the master stack.
    data = read_image(master_path).astype(np.float32)

    # Background correction (gradient removal).
    data = _background_correct(data)

    # Build output header.
    header = pyfits.Header()
    header["IMAGETYP"] = "Post-processed Light"
    header["STBLHASH"] = stack.stable_hash
    header["FILTER"] = stack.filter
    header["PIPELINE"] = "AstroFlow-OSS"

    if hasattr(stack, "center"):
        header["RA"] = stack.center[0]
        header["DEC"] = stack.center[1]

    # Write output.
    hdu = pyfits.PrimaryHDU(data=data, header=header)
    hdu.writeto(str(output_filename), overwrite=True)

    # Plate solving (modifies file in place).
    if not mock_pixinsight and hasattr(stack, "center"):
        _solve_field(
            output_filename,
            ra_hint=stack.center[0],
            dec_hint=stack.center[1],
            scale_hint=stack.image_scale,
        )

    log.info("OSS post-processing complete", output=str(output_filename))
    return WithSidecar(FitsFilePath(output_filename))
