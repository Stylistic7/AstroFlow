import os
from pathlib import Path
from astropy.time import Time

from file_io.generic import read_image, read_header, copy_image
from file_io.types import XisfFilePath
from pix import PixInsight
from pix.classes import ImageList
from pix.tasks import (
    ChannelCombination,
    SpectrophotometricColorCalibration,
    ImageSolver,
    StarXTerminator,
    LinearFit,
    SaveImages,
)
from context import context
from decorators import cached, create_thumbnail, measure_performance
from previews.types import QualityLevel
from mosaics.final_storage import store_mosaic_color_channel_master
from mosaics.types import Mosaic
from mosaics.artifacts import mosaic_filename_xisf
from products import makes_image


@PixInsight(slots=[1])
def _post_process_mosaic(
    images: ImageList, mosaic: Mosaic, output_filename: Path
):
    print(f"### _post_process_mosaic mosaic {mosaic} images {images}")

    if "RGB" in images:
        rgb = ChannelCombination(images.channels("RGB"))
        julian_date = int(Time(mosaic.oldest_date).jd)
        rgb = ImageSolver(
            rgb,
            metadata_ra=mosaic.center[0],
            metadata_dec=mosaic.center[1],
            metadata_resolution=mosaic.image_scale / 3600.0,
            metadata_observationTime=julian_date,
            metadata_epoch=julian_date,
        )
        rgb = SpectrophotometricColorCalibration(rgb)
        stars, _ = StarXTerminator(rgb)

    if "SHO" in images:
        _, s = StarXTerminator(images.channels("s"))
        _, h = StarXTerminator(images.channels("h"))
        _, o = StarXTerminator(images.channels("o"))
        s, o = LinearFit([s, o], h)
        narrowband = ChannelCombination([s, h, o], "SHO")

        # TODO: Supply multiple output filenames.
        SaveImages([narrowband], [output_filename])


@makes_image(Mosaic).as_post_processed(mosaic_filename_xisf)
@cached.remote(resources={"PixInsight": 1})
@measure_performance
@create_thumbnail
def post_process_mosaic(mode: QualityLevel, mosaic: Mosaic) -> XisfFilePath | None:
    print(f"### Entering post_process_mosaic for {mode=} mosaic {str(mosaic)}")

    if mode != QualityLevel.MASTER:
        return None

    filters = set()
    for target_id in mosaic.target_ids:
        for stack_id in context.targets[target_id].stack_ids:
            filters.add(context.stacks[stack_id].filter)
    filters = tuple(filters)

    images = [
        store_mosaic_color_channel_master.filename_fn(mosaic, filter)
        for filter in filters
    ]

    print(f"### post_process_mosaic {filters=} {images=}")

    for image in images:
        print(
            f"### post_process_mosaic image {image} "
            f"type {type(image)} exists {os.path.exists(image)}"
        )

    output_filename = post_process_mosaic.filename_fn(mosaic, None)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"### post_process_mosaic output_filename {output_filename}")

    if context.settings and context.settings.development_settings.mock_pixinsight:
        print(
            f"### post_process_mosaic mocking PixInsight "
            f"{images[0]} -> {output_filename}"
        )
        copy_image(images[0], output_filename)
    else:
        _post_process_mosaic(images, mosaic, output_filename)

    return output_filename
