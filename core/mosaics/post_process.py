"""Mosaic post-processing — TODO.

Pre-refactor this composed per-filter mosaic channels into an RGB /
SHO post-processed master. In the new architecture pipelines should
compose this themselves by passing per-filter `mosaic_channel`
outputs as data deps to a `combine_mosaic` op. Not in the active
pipeline trace path yet.
"""
from pathlib import Path

import structlog

from pix import Image, ImageList
from pix.tasks import (
    ChannelCombination,
    SpectrophotometricColorCalibration,
    WriteAstrometricSolution,
    StarXTerminator,
    LinearFit,
)
import plate_solving
from file_io.types import XisfFilePath, WithSidecar
from decorators import node
from mosaics.types import Mosaic

log = structlog.get_logger()


def _post_process_mosaic(
    images: ImageList, mosaic: Mosaic, wcs_header: dict,
) -> Image | None:
    if "RGB" in images:
        rgb = ChannelCombination(images.channels("RGB"))
        rgb = WriteAstrometricSolution(rgb, wcs_header)
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
    display="Combined",
    resources={"PixInsight": 1},
    auto_memory=True,
    plate_solve=False,  # RGB/SHO channel combination — not a mono star field
)
def post_process_mosaic(
    mosaic: Mosaic,
    channels,  # dict[str, WithSidecar]: per-filter mosaic-channel masters
    mock_pixinsight: bool,
) -> WithSidecar[XisfFilePath] | None:
    """Combine per-filter mosaic channels into an RGB / SHO master.

    `channels` is a dict keyed by filter, values are upstream
    WithSidecar handles produced by `post_process_mosaic_color_channel`.
    The pipeline body builds the dict and passes it as a data dep."""
    filter_list = sorted(channels.keys())
    image_paths = [channels[f].filename for f in filter_list]
    images = Image.open_many(image_paths, filters=filter_list)

    # Each per-filter mosaic-channel master already has a solve-field plate
    # solution in its sidecar (`compute_image_metadata` runs after every
    # produced artifact via the @makes_image trace hook). ChannelCombination
    # of mono channels inherits WCS from the active view (first channel),
    # so the first channel's solution IS the combined RGB's solution.
    ps = plate_solving.solve_field(Path(image_paths[0]))
    if ps is None:
        raise ValueError(
            f"post_process_mosaic: first channel master has no plate "
            f"solution: {image_paths[0]}"
        )

    result = _post_process_mosaic(images, mosaic, ps.header)
    if result is None:
        return None

    output_filename = post_process_mosaic.filename_fn(mosaic, channels, mock_pixinsight)
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    result.save_to(output_filename, mock=mock_pixinsight)
    return WithSidecar(output_filename)
