"""Mosaic post-processing — TODO.

Pre-refactor this composed per-filter mosaic channels into an RGB /
SHO post-processed master. In the new architecture pipelines should
compose this themselves by passing per-filter `mosaic_channel`
outputs as data deps to a `combine_mosaic` op. Not in the active
pipeline trace path yet.
"""
import structlog

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
from mosaics.types import Mosaic
from products.identity import MosaicId
from astropy.time import Time

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


@node(
    identity=MosaicId,
    resources={"PixInsight": 1},
    auto_memory=True,
)
def post_process_mosaic(
    mosaic: Mosaic,
    channels,  # dict[str, WithSidecar]: per-filter mosaic-channel masters
    mock_pixinsight: bool,
) -> WithSidecar[XisfFilePath] | None:
    """Combine per-filter mosaic channels into an RGB / SHO master.

    `channels` is a dict keyed by filter, values are upstream
    WithSidecar handles produced by `store_mosaic_color_channel_xisf`.
    The pipeline body builds the dict and passes it as a data dep."""
    filter_list = sorted(channels.keys())
    image_paths = [channels[f].filename for f in filter_list]
    images = Image.open_many(image_paths, filters=filter_list)

    result = _post_process_mosaic(images, mosaic)
    if result is None:
        return None

    output_filename = post_process_mosaic.filename_fn(mosaic, channels, mock_pixinsight)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    result.save_to(output_filename, mock=mock_pixinsight)
    return WithSidecar(output_filename)
