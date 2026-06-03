"""Target post-processing — TODO.

Pre-refactor this composed per-filter target channels into a final
target master (RGB / SHO). Not in the active pipeline trace path
yet; the new architecture would have a pipeline body wire the
per-filter `merge_target_channel` outputs as data deps.
"""
from pathlib import Path

from pix import Image, ImageList
from pix.tasks import (
    ChannelCombination,
    SpectrophotometricColorCalibration,
    WriteAstrometricSolution,
    StarXTerminator,
    LinearFit,
    StarAlignment,
    LRGBCombination,
)
import plate_solving
from file_io.types import XisfFilePath, WithSidecar
from decorators import node
from targets.types import Target
from products.identity import TargetId


def _post_process_target(
    images: ImageList, target: Target, wcs_header: dict,
) -> tuple[Image, Image] | None:
    if "LRGB" in images:
        aligned = StarAlignment(images.channels("LRGB"))
        rgb = LRGBCombination([aligned[i] for i in range(len(aligned))])
        solved = WriteAstrometricSolution(rgb, wcs_header)
        calibrated = SpectrophotometricColorCalibration(solved, saturationThreshold=1.0)
        stars, starless = StarXTerminator(calibrated)
        return stars, starless

    if "RGB" in images:
        aligned = StarAlignment(images.channels("RGB"))
        rgb = ChannelCombination(aligned)
        solved = WriteAstrometricSolution(rgb, wcs_header)
        calibrated = SpectrophotometricColorCalibration(solved)
        stars, starless = StarXTerminator(calibrated)
        return stars, starless

    if "SHO" in images:
        aligned = StarAlignment(images.channels("SHO"))
        _, s = StarXTerminator(aligned[0])
        _, h = StarXTerminator(aligned[1])
        _, o = StarXTerminator(aligned[2])
        s, o = LinearFit([s, o], h)
        narrowband = ChannelCombination([s, h, o], "SHO")
        return narrowband, None

    return None


@node(
    identity=TargetId,
    resources={"PixInsight": 1},
    scheduling_strategy="SPREAD",
    auto_memory=True,
)
def post_process_target(
    target: Target,
    channels,  # dict[str, WithSidecar]: per-filter channel masters
    mock_pixinsight: bool = False,
) -> WithSidecar[XisfFilePath] | None:
    """Combine per-filter target channels into a single target master.

    `channels` is a dict keyed by filter, values are upstream
    WithSidecar handles produced by `merge_target_channel`. The
    pipeline body builds the dict and passes it as a data dep —
    no filename_fn lookups needed."""
    filter_list = sorted(channels.keys())
    image_paths = [channels[f].filename for f in filter_list]
    images = Image.open_many(image_paths, filters=filter_list)

    # Each per-filter channel master already has a solve-field plate solution
    # in its sidecar (`compute_image_metadata` runs after every produced
    # artifact via the @makes_image trace hook). StarAlignment registers all
    # channels to the first one as reference, and ChannelCombination /
    # LRGBCombination inherit WCS from the active view — so the first
    # channel's solution IS the combined image's solution.
    ps = plate_solving.solve_field(Path(image_paths[0]))
    if ps is None:
        raise ValueError(
            f"post_process_target: first channel master has no plate "
            f"solution: {image_paths[0]}"
        )

    result = _post_process_target(images, target, ps.header)
    if result is None:
        return None

    output_filename = post_process_target.filename_fn(target, channels, mock_pixinsight)
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    stars, starless = result
    # We only have one filename_fn output but two artifacts. Drop the
    # stars layer for now (the starless body is what mosaics consume);
    # if a workflow needs both, split into two @node ops.
    if starless is not None:
        starless.save_to(output_filename, mock=mock_pixinsight)
        return WithSidecar(output_filename)
    if stars is not None:
        stars.save_to(output_filename, mock=mock_pixinsight)
        return WithSidecar(output_filename)
    return None


@node(identity=TargetId)
def post_process_target_oss(target: Target, mock_pixinsight: bool = False) -> None:
    """Stub: target post-processing for the OSS pipeline (TODO)."""
    return None
