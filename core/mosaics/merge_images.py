import os
import ray
import numpy as np
import structlog
from beartype import beartype
from pathlib import Path
from dataclasses import dataclass

from file_io.generic import read_header, read_image
from pix import Image
from pix.tasks import PhotometricMosaic
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from previews.types import QualityLevel, ALL_IMPLEMENTATIONS, PIXINSIGHT, SUBFRAME, MASTER, DOWNSCALED_MASTER
from decorators import node, create_tile_pyramid
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from mosaics.artifacts import merged_filename_tiff, merged_filename_xisf
from products import makes_image
from products.identity import MergeProgressId, PresentationType

THUMBNAIL = PresentationType.THUMBNAIL
from products.registry import get_filename_fn, MediaType
from hashing import hash_list

log = structlog.get_logger()


@dataclass(frozen=True)
class MergedImage:
    pass


def _merge_tile_path(mode, mosaic, filter, mock_pixinsight, input_0, input_1):
    """Fixed tile pyramid location per (mosaic, filter, mode). Each merge
    overwrites the previous, so the web server always serves the latest."""
    from context import context
    return context.output_dir / "Mosaic" / str(mode) / f"progress_{mosaic.stable_hash}_{filter}.tiff"



@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(MergedImage, ALL_IMPLEMENTATIONS, [SUBFRAME], merged_filename_tiff,
             identity=MergeProgressId, presentations=[THUMBNAIL])
@makes_image(MergedImage, PIXINSIGHT, [MASTER, DOWNSCALED_MASTER], merged_filename_xisf,
             identity=MergeProgressId, presentations=[THUMBNAIL])
@create_tile_pyramid(tile_filename=_merge_tile_path)
def merge_images(
    mode: QualityLevel,
    mosaic: Mosaic,
    filter: str,
    mock_pixinsight: bool,
    input_0: ImageMetadata,
    input_1: ImageMetadata,
) -> WithSidecar:
    path_0 = input_0.filename.resolve(strict=True)
    path_1 = input_1.filename.resolve(strict=True)

    output_fn = get_filename_fn(MediaType.IMAGE, MergedImage, mode)
    output_filename = output_fn(mosaic, input_0, input_1)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    if mode == SUBFRAME:
        stable_hash = hash_list([input_0.stable_hash, input_1.stable_hash])
        filters = set(read_header(i)["FILTER"] for i in [path_0, path_1])
        assert len(filters) == 1
        filter = next(iter(filters))

        image_0 = read_image(path_0)
        image_1 = read_image(path_1)

        assert not np.isnan(np.min(image_0))
        assert not np.isnan(np.min(image_1))
        overlap = np.where((image_0 > 0) & (image_1 > 0))
        output_image = np.maximum(image_0, image_1)
        output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
        assert not np.isnan(np.min(output_image))

        write_tiff_image(Path(output_filename), output_image, {"FILTER": filter, "STBLHASH": stable_hash})
    else:
        images = Image.open_many([path_0, path_1])
        result = PhotometricMosaic(images)
        result.save_to(output_filename, mock=mock_pixinsight)

    log.info("Merge images complete", output=output_filename)
    return WithSidecar(output_filename)
