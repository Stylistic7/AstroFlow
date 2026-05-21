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
from decorators import node
from previews.types import PIXINSIGHT, OSS, MASTER,POST_PROCESSED
from mosaics.types import Mosaic
from products import makes_image
from products.identity import MosaicId, MosaicChannelId, PresentationType

THUMBNAIL = PresentationType.THUMBNAIL
TILE_PYRAMID = PresentationType.TILE_PYRAMID
from products.registry import get_filename_fn, MediaType
from targets.types import TargetsDict
from targets.target_channel import filters_for_target

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


@node(resources={"PixInsight": 1}, auto_memory=True)
@makes_image(PIXINSIGHT, [POST_PROCESSED],
             identity=MosaicId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath)
def post_process_mosaic(
    mode: str,
    mosaic: Mosaic,
    targets: TargetsDict,
) -> WithSidecar[XisfFilePath] | None:
    from context import context
    log.info("Entering post_process_mosaic", mode=mode, mosaic=str(mosaic))

    if mode != MASTER:
        return None

    filters: set[str] = set()
    for target_id in mosaic.target_ids:
        filters.update(filters_for_target(targets[target_id].get()))
    # Pin iteration order so the parallel filters list lines up with the paths.
    filter_list = list(filters)

    master_fn = get_filename_fn(MediaType.IMAGE, MosaicChannelId, MASTER)
    image_paths = [master_fn(mosaic, f) for f in filter_list]

    images = Image.open_many(image_paths, filters=filter_list)
    result = _post_process_mosaic(images, mosaic)
    if result is None:
        return None

    output_filename = post_process_mosaic.filename_fn(mosaic, None)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    mock = context.settings.development_settings.mock_pixinsight
    result.save_to(output_filename, mock=mock)

    return WithSidecar(output_filename)


@node
@makes_image(OSS, [POST_PROCESSED],
             identity=MosaicId, presentations=[THUMBNAIL, TILE_PYRAMID],
             file_type=XisfFilePath)
def post_process_mosaic_oss(
    mode: str,
    mosaic: Mosaic,
    targets: TargetsDict,
) -> None:
    """Stub: mosaic post-processing for the OSS pipeline (TODO)."""
    return None
