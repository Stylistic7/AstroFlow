import os
import structlog
from pathlib import Path
from astropy.time import Time

from pix import Image, ImageList
from pix.tasks import (
    ChannelCombination,
    SpectrophotometricColorCalibration,
    ImageSolver,
    StarXTerminator,
    LinearFit,
)
from file_io.types import XisfFilePath, WithSidecar
from decorators import node, create_thumbnail
from previews.types import QualityLevel
from mosaics.final_storage import store_mosaic_color_channel_master
from mosaics.types import Mosaic
from mosaics.artifacts import mosaic_filename_xisf
from products import makes_image
from stacks.composite_stack import CompositeStacksDict
from targets.types import TargetsDict

log = structlog.get_logger()


def _post_process_mosaic(images: ImageList, mosaic: Mosaic) -> Image | None:
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
        stars, starless = StarXTerminator(rgb)
        return stars

    if "SHO" in images:
        _, s = StarXTerminator(images["S"])
        _, h = StarXTerminator(images["H"])
        _, o = StarXTerminator(images["O"])
        s, o = LinearFit([s, o], h)
        narrowband = ChannelCombination([s, h, o], "SHO")
        return narrowband

    return None


@makes_image(Mosaic).as_post_processed(mosaic_filename_xisf)
@node(resources={"PixInsight": 1})
@create_thumbnail
def post_process_mosaic(
    mode: QualityLevel,
    mosaic: Mosaic,
    stacks: CompositeStacksDict,
    targets: TargetsDict,
) -> WithSidecar[XisfFilePath] | None:
    from context import context
    log.info("Entering post_process_mosaic", mode=mode, mosaic=str(mosaic))

    if mode != QualityLevel.MASTER:
        return None

    filters = set()
    for target_id in mosaic.target_ids:
        for stack_id in targets[target_id].get().stack_ids:
            filters.add(stacks[stack_id].get().filter)

    image_paths = [
        store_mosaic_color_channel_master.filename_fn(mosaic, f)
        for f in filters
    ]

    images = Image.open_many(image_paths)
    result = _post_process_mosaic(images, mosaic)
    if result is None:
        return None

    output_filename = post_process_mosaic.filename_fn(mosaic, None)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    mock = context.settings.development_settings.mock_pixinsight
    result.save_to(output_filename, mock=mock)

    return WithSidecar(output_filename)
