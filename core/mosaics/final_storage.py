import ray
import shutil
from pathlib import Path
import numpy as np
from beartype import beartype

from mosaics.types import Mosaic
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from pix import Image
from products import makes_image
from products.identity import MosaicChannelId, PresentationType
from decorators import node

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID
from file_io.types import FitsFilePath, TiffFilePath, XisfFilePath, WithSidecar
from metadata.classes import ImageMetadata


def _store_mosaic_color_channel_pi(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool,
    input: ImageMetadata, output: XisfFilePath,
) -> WithSidecar[XisfFilePath]:
    """Shared body for PI-backed mosaic-channel stores. The XISF file
    that lands at `output` is the canonical "this pipeline's per-
    filter mosaic" artifact — the dashboard's `Mosaic` tap points
    at it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(input.filename, filter=filter)
    image.save_to(output, mock=mock_pixinsight)
    return WithSidecar(output)


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=MosaicChannelId, display="Mosaic")
def store_mosaic_color_channel_master(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool,
    input: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """Full-resolution per-filter mosaic master (PixInsight Master)."""
    return _store_mosaic_color_channel_pi(
        mosaic, filter, mock_pixinsight, input,
        store_mosaic_color_channel_master.filename_fn(mosaic, filter),
    )


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(identity=MosaicChannelId, display="Mosaic")
def store_mosaic_color_channel_downscaled_master(
    mosaic: Mosaic, filter: str, mock_pixinsight: bool,
    input: ImageMetadata,
) -> WithSidecar[XisfFilePath]:
    """Downscaled-resolution per-filter mosaic master (PI Downscaled)."""
    return _store_mosaic_color_channel_pi(
        mosaic, filter, mock_pixinsight, input,
        store_mosaic_color_channel_downscaled_master.filename_fn(mosaic, filter),
    )


@node(resources={"fastest_storage": 0.001})
@makes_image(identity=MosaicChannelId, display="Subframe")
def store_mosaic_color_channel_subframe(
    mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    filename = input.filename
    assert filename.exists()
    output = store_mosaic_color_channel_subframe.filename_fn(mosaic, filter)
    image = read_image(filename)
    autostretched = (autostretch(image) * 255.0).astype(np.uint8)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_tiff_image(output, autostretched, {"FILTER": filter}, tile=(256, 256))
    return WithSidecar(output)


def _store_mosaic_color_channel_oss(
    mosaic: Mosaic, filter: str, input: ImageMetadata, output: FitsFilePath,
) -> WithSidecar[FitsFilePath]:
    """Shared body for OSS-backed mosaic-channel stores. Just copies
    the merge output (already a FITS) to the canonical mosaic path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(input.filename), str(output))
    return WithSidecar(FitsFilePath(output))


@node
@makes_image(identity=MosaicChannelId, display="Mosaic")
def store_mosaic_color_channel_oss_master(
    mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """Full-resolution OSS mosaic-channel store."""
    return _store_mosaic_color_channel_oss(
        mosaic, filter, input,
        store_mosaic_color_channel_oss_master.filename_fn(mosaic, filter),
    )


@node
@makes_image(identity=MosaicChannelId, display="Mosaic")
def store_mosaic_color_channel_oss_downscaled_master(
    mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar[FitsFilePath]:
    """Downscaled-resolution OSS mosaic-channel store."""
    return _store_mosaic_color_channel_oss(
        mosaic, filter, input,
        store_mosaic_color_channel_oss_downscaled_master.filename_fn(mosaic, filter),
    )
