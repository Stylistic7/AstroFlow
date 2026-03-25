import os
import ray
import numpy as np
from beartype import beartype
from pathlib import Path
from dataclasses import dataclass

from file_io.generic import read_header, read_image
from pix import Image
from pix.tasks import PhotometricMosaic
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from mosaics.artifacts import merged_filename_tiff, merged_filename_xisf
from previews.types import QualityLevel
from decorators import node, measure_performance
from file_io.tiff_file_io import write_tiff_image
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from products import makes_image
from context import context


@dataclass(frozen=True)
class MergedImage:
    pass


@makes_image(MergedImage).as_subframe(merged_filename_tiff)
@node
def merge_images_with_maximum(
    mosaic: Mosaic, input_0: ImageMetadata, input_1: ImageMetadata
) -> WithSidecar[TiffFilePath]:
    input_0 = input_0.filename.resolve(strict=True)
    input_1 = input_1.filename.resolve(strict=True)

    filters = set(read_header(i)["FILTER"] for i in [input_0, input_1])
    assert len(filters) == 1
    filter = next(iter(filters))

    image_0 = read_image(input_0)
    image_1 = read_image(input_1)

    assert not np.isnan(np.min(image_0))
    assert not np.isnan(np.min(image_1))
    overlap = np.where((image_0 > 0) & (image_1 > 0))
    output_image = np.maximum(image_0, image_1)
    output_image[overlap] = (image_0[overlap] + image_1[overlap]) / 2.0
    assert not np.isnan(np.min(output_image))

    output_filename = merge_images_with_maximum.filename_fn(
        mosaic, input_0, input_1
    )
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    write_tiff_image(Path(output_filename), output_image, {"FILTER": filter})

    return WithSidecar(output_filename)


@makes_image(MergedImage).as_master(merged_filename_xisf)
@node
def merge_images_with_pmm(
    mosaic: Mosaic, input_0: ImageMetadata, input_1: ImageMetadata
) -> WithSidecar[XisfFilePath]:
    input_0_path = input_0.filename.resolve(strict=True)
    input_1_path = input_1.filename.resolve(strict=True)

    output_filename = merge_images_with_pmm.filename_fn(
        mosaic, input_0_path, input_1_path
    )
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    mock = context.settings.development_settings.mock_pixinsight

    images = Image.open_many([input_0_path, input_1_path])
    result = PhotometricMosaic(images)
    result.save_to(output_filename, mock=mock)

    return WithSidecar(output_filename)


@ray.remote
@beartype
@measure_performance
def merge_images(
    mode: QualityLevel,
    mosaic: Mosaic,
    input_0: ImageMetadata,
    input_1: ImageMetadata,
) -> WithSidecar:
    if mode == QualityLevel.SUBFRAME:
        result = ray.get(
            merge_images_with_maximum.remote(mosaic, input_0, input_1)
        )
    else:
        result = ray.get(merge_images_with_pmm.remote(mosaic, input_0, input_1))

    print(f"### merge_images {result=}")
    return result
