"""OSS mosaic merging: distance-weighted blending with flux equalization.

Replaces PixInsight's PhotometricMosaic. The placed images are already
in the same coordinate system (the mosaic viewbox), so this is purely
a blending operation — no WCS reprojection needed.
"""
import os
import numpy as np
import structlog
from pathlib import Path

import astropy.io.fits as pyfits
from scipy.ndimage import distance_transform_edt

from decorators import node, create_tile_pyramid
from file_io.generic import read_image, read_header
from file_io.types import FitsFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from mosaics.artifacts import merged_filename_fits
from mosaics.merge_images import MergedImage, _merge_tile_path
from previews.types import QualityLevel, OSS, MASTER, DOWNSCALED_MASTER
from products import makes_image
from products.registry import get_filename_fn, MediaType
from hashing import hash_list

log = structlog.get_logger()


def _blend_images(image_0, image_1):
    """Distance-weighted blend of two overlapping images with flux equalization."""
    mask_0 = image_0 > 0
    mask_1 = image_1 > 0
    overlap = mask_0 & mask_1

    if not overlap.any():
        # No overlap — just combine non-overlapping regions.
        return np.maximum(image_0, image_1)

    # Flux equalization: scale image_1 to match image_0 in overlap region.
    mean_0 = np.mean(image_0[overlap])
    mean_1 = np.mean(image_1[overlap])
    if mean_1 > 0 and mean_0 > 0:
        scale = mean_0 / mean_1
        # Clamp scale to avoid extreme corrections.
        scale = np.clip(scale, 0.5, 2.0)
        image_1 = image_1 * scale

    # Distance weights from each image's edges.
    # Pixels deep inside an image get high weight; pixels near the edge
    # get low weight. This creates a smooth gradient across the seam.
    weight_0 = distance_transform_edt(mask_0).astype(np.float32)
    weight_1 = distance_transform_edt(mask_1).astype(np.float32)

    # Combine using distance weights.
    total_weight = weight_0 + weight_1
    total_weight[total_weight == 0] = 1.0  # avoid division by zero
    output = (weight_0 * image_0 + weight_1 * image_1) / total_weight

    return output


@node
@makes_image(MergedImage, OSS, [MASTER, DOWNSCALED_MASTER], merged_filename_fits)
@create_tile_pyramid(tile_filename=_merge_tile_path)
def merge_images_oss(
    mode: QualityLevel,
    mosaic: Mosaic,
    filter: str,
    mock_pixinsight: bool,
    input_0: ImageMetadata,
    input_1: ImageMetadata,
) -> WithSidecar:
    path_0 = input_0.filename.resolve(strict=True)
    path_1 = input_1.filename.resolve(strict=True)

    output_fn = get_filename_fn(MediaType.IMAGE, MergedImage, mode, OSS)
    output_filename = output_fn(mosaic, input_0, input_1)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    # Read both images.
    image_0 = read_image(path_0).astype(np.float32)
    image_1 = read_image(path_1).astype(np.float32)

    # Blend with distance-weighted feathering and flux equalization.
    result = _blend_images(image_0, image_1)

    # Write output as FITS.
    header = pyfits.Header()
    header["FILTER"] = filter
    header["STBLHASH"] = hash_list([input_0.stable_hash, input_1.stable_hash])
    header["PIPELINE"] = "AstroFlow-OSS"

    hdu = pyfits.PrimaryHDU(data=result.astype(np.float32), header=header)
    hdu.writeto(str(output_filename), overwrite=True)

    # Track latest merge for the web server progress display.
    from context import context
    context[("latest_merge", mosaic.stable_hash, filter, mode)] = str(output_filename)

    log.info("OSS merge complete", output=str(output_filename))
    return WithSidecar(output_filename)
