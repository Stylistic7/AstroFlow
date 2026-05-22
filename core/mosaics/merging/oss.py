"""OSS mosaic merging: distance-weighted blending with flux equalization.

Replaces PhotometricMosaic. Placed images are already in the same
coordinate system (the mosaic viewbox), so no WCS reprojection needed —
just feather across the seam and equalize brightness in the overlap.
"""
import numpy as np
import structlog
from pathlib import Path

import astropy.io.fits as pyfits
from scipy.ndimage import distance_transform_edt

from decorators import node
from file_io.generic import read_image
from file_io.types import FitsFilePath, WithSidecar
from hashing import hash_list
from metadata.classes import ImageMetadata
from mosaics.artifacts import merged_filename_fits
from mosaics.merging.pixinsight import (
    MergedImage, _panel_count, _set_panel_count,
)
from mosaics.types import Mosaic
from products import makes_image
from products.identity import MergedImageId

log = structlog.get_logger()

def _blend(image_0, image_1):
    """Distance-weighted blend with flux equalization in the overlap."""
    mask_0 = image_0 > 0
    mask_1 = image_1 > 0
    overlap = mask_0 & mask_1
    if not overlap.any():
        return np.maximum(image_0, image_1)

    # Match image_1's brightness to image_0 in the overlap region.
    mean_0 = np.mean(image_0[overlap])
    mean_1 = np.mean(image_1[overlap])
    if mean_0 > 0 and mean_1 > 0:
        image_1 = image_1 * np.clip(mean_0 / mean_1, 0.5, 2.0)

    # Feather: each pixel's weight is its distance from the image's edge,
    # so pixels deep inside an image dominate, pixels near an edge blend.
    w_0 = distance_transform_edt(mask_0).astype(np.float32)
    w_1 = distance_transform_edt(mask_1).astype(np.float32)
    total = w_0 + w_1
    total[total == 0] = 1.0
    return (w_0 * image_0 + w_1 * image_1) / total

@node(auto_memory=True)
@makes_image(filename_fn=merged_filename_fits, identity=MergedImageId)
def merge_images_oss(
    mosaic: Mosaic, filter: str,
    input_0: ImageMetadata, input_1: ImageMetadata,
) -> WithSidecar:
    """Distance-weighted blend of two placed-image inputs (OSS path).
    Pipeline-agnostic: PI Master and PI Downscaled both call this
    with their own placed inputs; the inputs' content_hashes differ,
    so `merged_filename_fits` lands them in different files."""
    output_filename = merge_images_oss.filename_fn(mosaic, filter, input_0, input_1)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    image_0 = read_image(input_0.filename.resolve(strict=True)).astype(np.float32)
    image_1 = read_image(input_1.filename.resolve(strict=True)).astype(np.float32)
    result = _blend(image_0, image_1)

    header = pyfits.Header()
    header["FILTER"] = filter
    header["STBLHASH"] = hash_list([input_0.stable_hash, input_1.stable_hash])
    header["PIPELINE"] = "AstroFlow-OSS"
    pyfits.PrimaryHDU(data=result.astype(np.float32), header=header).writeto(
        str(output_filename), overwrite=True,
    )

    new_count = _panel_count(input_0) + _panel_count(input_1)
    _set_panel_count(output_filename, new_count)

    log.info("OSS merge complete", output=str(output_filename), panel_count=new_count)
    return WithSidecar(output_filename)
