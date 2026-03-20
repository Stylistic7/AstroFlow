import ray
from pathlib import Path
import numpy as np
import tifffile
from beartype import beartype

from context import context
from mosaics.artifacts import mosaic_filename_tiff, mosaic_filename_xisf
from file_io.generic import read_image, copy_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from pix2 import Image
from products import makes_image
from decorators import cached, create_thumbnail, measure_performance
from file_io.types import TiffFilePath, XisfFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic
from previews.types import QualityLevel


def make_preview_tiff(source_path: Path, dest_path: Path):
    """Read image and save to .tif"""
    assert source_path != dest_path
    arr = autostretch(read_image(source_path))
    arr = (arr * 255.0).astype(np.uint8)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(dest_path, arr, tile=(256, 256), compression="deflate")


@makes_image(Mosaic).as_master(mosaic_filename_xisf)
@cached.remote(resources={"PixInsight": 1})
@create_thumbnail
@measure_performance
def store_mosaic_color_channel_master(
    mosaic: Mosaic, filter: str, input: ImageMetadata
) -> WithSidecar[XisfFilePath]:
    filename = input.filename
    output = store_mosaic_color_channel_master.filename_fn(mosaic, filter)
    tiff_output = output.with_suffix(".tiff")
    output.parent.mkdir(parents=True, exist_ok=True)

    mock = context.settings.development_settings.mock_pixinsight

    # Save via pix2: opens the image in PixInsight and saves as uncompressed XISF
    image = Image.open(filename)
    image.save_to(output, mock=mock)
    make_preview_tiff(output if not mock else filename, tiff_output)

    return WithSidecar(output)


@makes_image(Mosaic).as_subframe(mosaic_filename_tiff)
@cached.remote(resources={"fastest_storage": 0.001})
@create_thumbnail
@measure_performance
def store_mosaic_color_channel_subframe(
    mosaic: Mosaic, filter: str, input: ImageMetadata
) -> WithSidecar[TiffFilePath]:
    filename = input.filename
    print(
        f"### store_mosaic_color_channel_subframe type {str(mosaic)} {filter=} {filename=}"
    )

    assert filename.exists()

    output = store_mosaic_color_channel_subframe.filename_fn(mosaic, filter)
    image = read_image(filename)
    autostretched = (autostretch(image) * 255.0).astype(np.uint8)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_tiff_image(output, autostretched, {"FILTER": filter}, tile=(256, 256))

    return WithSidecar(output)


@ray.remote
@beartype
def store_mosaic_color_channel(
    mode: QualityLevel, mosaic: Mosaic, filter: str, input: ImageMetadata
) -> WithSidecar:
    print(
        f"### store_mosaic_color_channel {mode=} {str(mosaic)} {filter=} {input=}"
    )
    if mode == QualityLevel.SUBFRAME:
        return ray.get(
            store_mosaic_color_channel_subframe.remote(mosaic, filter, input)
        )
    else:
        return ray.get(
            store_mosaic_color_channel_master.remote(mosaic, filter, input)
        )
