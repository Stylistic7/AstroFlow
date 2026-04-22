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

from decorators import node, create_tile_pyramid
from file_io.generic import read_image
from file_io.types import FitsFilePath, WithSidecar
from hashing import hash_list
from metadata.classes import ImageMetadata
from mosaics.artifacts import merged_filename_fits
from mosaics.merge_images import MergedImage, _merge_tile_path
from mosaics.types import Mosaic
from previews.types import QualityLevel, OSS, MASTER, DOWNSCALED_MASTER
from products import makes_image
from products.identity import MergeProgressId, PresentationType

THUMBNAIL = PresentationType.THUMBNAIL
from products.registry import get_filename_fn, MediaType

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
@makes_image(MergedImage, OSS, [MASTER, DOWNSCALED_MASTER], merged_filename_fits,
             identity=MergeProgressId, presentations=[THUMBNAIL])
@create_tile_pyramid(tile_filename=_merge_tile_path)
def merge_images_oss(
    mode: QualityLevel,
    mosaic: Mosaic,
    filter: str,
    mock_pixinsight: bool,
    input_0: ImageMetadata,
    input_1: ImageMetadata,
) -> WithSidecar:
    output_filename = get_filename_fn(MediaType.IMAGE, MergedImage, mode, OSS)(
        mosaic, input_0, input_1,
    )
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

    from context import context
    context[("latest_merge", mosaic.stable_hash, filter, mode)] = str(output_filename)
    log.info("OSS merge complete", output=str(output_filename))
    return WithSidecar(output_filename)
