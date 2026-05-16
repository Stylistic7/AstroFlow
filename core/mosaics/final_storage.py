import ray
import shutil
from pathlib import Path
import numpy as np
from beartype import beartype

from mosaics.types import Mosaic
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from previews.types import QualityLevel, Implementation, ALL_IMPLEMENTATIONS, PIXINSIGHT, OSS, SUBFRAME, MASTER, DOWNSCALED_MASTER
from pix import Image
from products import makes_image
from products.identity import MosaicChannelId, PresentationType
from decorators import node

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID
from file_io.types import FitsFilePath, TiffFilePath, XisfFilePath, WithSidecar
from metadata.classes import ImageMetadata


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(PIXINSIGHT, [MASTER, DOWNSCALED_MASTER],
             identity=MosaicChannelId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath)
def store_mosaic_color_channel_xisf(
    mode: QualityLevel, mosaic: Mosaic, filter: str, input: ImageMetadata,
    mock_pixinsight: bool = False,
) -> WithSidecar[XisfFilePath]:
    fn = store_mosaic_color_channel_xisf.filename_fns[(MosaicChannelId, mode)]
    filename = input.filename
    output = fn(mosaic, filter)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(filename)
    image.save_to(output, mock=mock_pixinsight)

    return WithSidecar(output)


@node(resources={"fastest_storage": 0.001})
@makes_image(ALL_IMPLEMENTATIONS, [SUBFRAME],
             identity=MosaicChannelId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=TiffFilePath)
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


@node
@makes_image(OSS, [MASTER, DOWNSCALED_MASTER],
             identity=MosaicChannelId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=FitsFilePath)
def store_mosaic_color_channel_oss(
    mode: QualityLevel, mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """Copy the final merged FITS to the canonical mosaic path."""
    fn = store_mosaic_color_channel_oss.filename_fns[(MosaicChannelId, mode)]
    output = fn(mosaic, filter)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input.filename), str(output))
    return WithSidecar(FitsFilePath(output))


@ray.remote
@beartype
def store_mosaic_color_channel(
    mode: QualityLevel, mosaic: Mosaic, filter: str,
    implementation: Implementation, input: ImageMetadata,
) -> WithSidecar:
    from context import context
    mock = context.settings.development_settings.mock_pixinsight
    if mode == SUBFRAME:
        return ray.get(
            store_mosaic_color_channel_subframe.remote(mosaic, filter, input)
        )
    if implementation == OSS:
        return ray.get(
            store_mosaic_color_channel_oss.remote(mode, mosaic, filter, input)
        )
    return ray.get(
        store_mosaic_color_channel_xisf.remote(mode, mosaic, filter, input, mock)
    )
