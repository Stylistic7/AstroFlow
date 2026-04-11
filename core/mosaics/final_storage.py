import ray
from pathlib import Path
import numpy as np
from beartype import beartype

from mosaics.artifacts import mosaic_filename_tiff, mosaic_filename_xisf
from mosaics.types import Mosaic
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from previews.operations import autostretch
from previews.types import QualityLevel, SUBFRAME, MASTER, DOWNSCALED_MASTER
from pix import Image
from products import makes_image
from decorators import node, create_thumbnail, create_tile_pyramid
from file_io.types import TiffFilePath, XisfFilePath, WithSidecar
from metadata.classes import ImageMetadata


@node(resources={"PixInsight": 1})
@makes_image(Mosaic).for_levels([MASTER, DOWNSCALED_MASTER], mosaic_filename_xisf)
@create_thumbnail
@create_tile_pyramid
def store_mosaic_color_channel_xisf(
    mode: QualityLevel, mosaic: Mosaic, filter: str, input: ImageMetadata,
    mock_pixinsight: bool = False,
) -> WithSidecar[XisfFilePath]:
    fn = store_mosaic_color_channel_xisf.filename_fns[(Mosaic, mode)]
    filename = input.filename
    output = fn(mosaic, filter)
    output.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(filename)
    image.save_to(output, mock=mock_pixinsight)

    return WithSidecar(output)


@node(resources={"fastest_storage": 0.001})
@makes_image(Mosaic).for_levels([SUBFRAME], mosaic_filename_tiff)
@create_thumbnail
@create_tile_pyramid
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
    mode: QualityLevel, mosaic: Mosaic, filter: str, input: ImageMetadata,
) -> WithSidecar:
    from context import context
    mock = context.settings.development_settings.mock_pixinsight
    if mode == SUBFRAME:
        return ray.get(
            store_mosaic_color_channel_subframe.remote(mosaic, filter, input)
        )
    else:
        return ray.get(
            store_mosaic_color_channel_xisf.remote(mode, mosaic, filter, input, mock)
        )
