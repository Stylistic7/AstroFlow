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
from mosaics.artifacts import merged_filename_tiff, merged_filename_xisf
from previews.types import QualityLevel, SUBFRAME, MASTER, DOWNSCALED_MASTER
from decorators import node
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from products import makes_image
from products.registry import get_filename_fn, MediaType
from hashing import hash_list

log = structlog.get_logger()


@dataclass(frozen=True)
class MergedImage:
    pass


@makes_image(MergedImage).for_levels([SUBFRAME], merged_filename_tiff)
@makes_image(MergedImage).for_levels([MASTER, DOWNSCALED_MASTER], merged_filename_xisf)
@node(resources={"PixInsight": 1})
def merge_images(
    mode: QualityLevel,
    mosaic: Mosaic,
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
