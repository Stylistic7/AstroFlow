import os
from pathlib import Path
from astropy.time import Time

from pix import Image, ImageList
from pix.tasks import (
    ChannelCombination,
    SpectrophotometricColorCalibration,
    ImageSolver,
    StarXTerminator,
    LinearFit,
    StarAlignment,
    LRGBCombination,
)
from context import context
from file_io.types import XisfFilePath
from decorators import node
from targets.types import Target
from targets.target_channel import TargetChannel, group_stacks_by_filter
from products import makes_image
from products.identity import TargetId, TargetChannelId, PresentationType

THUMBNAIL = PresentationType.THUMBNAIL
from previews.types import PIXINSIGHT, OSS, MASTER,POST_PROCESSED


def _post_process_target(images: ImageList, target: Target) -> tuple[Image, Image] | None:
    if "LRGB" in images:
        aligned = StarAlignment(images.channels("LRGB"))
        rgb = LRGBCombination([aligned[i] for i in range(len(aligned))])
        julian_date = int(Time(target.oldest_date).jd)
        solved = ImageSolver(
            rgb,
            metadata_ra=target.center[0],
            metadata_dec=target.center[1],
            metadata_resolution=target.image_scale / 3600.0,
            metadata_observationTime=julian_date,
            metadata_epoch=julian_date,
        )
        calibrated = SpectrophotometricColorCalibration(solved, saturationThreshold=1.0)
        stars, starless = StarXTerminator(calibrated)
        return stars, starless

    if "RGB" in images:
        aligned = StarAlignment(images.channels("RGB"))
        rgb = ChannelCombination(aligned)
        julian_date = int(Time(target.oldest_date).jd)
        solved = ImageSolver(
            rgb,
            metadata_ra=target.center[0],
            metadata_dec=target.center[1],
            metadata_resolution=target.image_scale / 3600.0,
            metadata_observationTime=julian_date,
            metadata_epoch=julian_date,
        )
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


def target_filename(target: Target, suffix: str) -> XisfFilePath:
    return XisfFilePath(f"target {target.stable_hash}_{suffix}.xisf")


def _target_channel_paths(
    target: Target, implementation
) -> tuple[list[Path], list[str]]:
    """Resolve the per-(target, filter) merged-master paths.

    Reads which filters this target has from `context.stacks` and looks
    up the registered TargetChannel filename for each. Returns parallel
    lists of (paths, filters) so callers can attach filter metadata to
    the lazy Images without re-reading anything from disk.
    """
    stacks = context.stacks
    target_stacks = [stacks[sid].get() for sid in target.stack_ids]
    by_filter = group_stacks_by_filter(target_stacks)
    # Per-filter channel masters come from `merge_target_channel`
    # for PixInsight or `merge_target_channel_oss` for OSS.
    if implementation == OSS:
        from targets.merging.oss import merge_target_channel_oss as op
    else:
        from targets.merging.pixinsight import merge_target_channel as op
    channel_fn = op.filename_fns[(TargetChannelId, MASTER)]
    filters = list(by_filter.keys())
    paths = [Path(channel_fn(target, f)) for f in filters]
    return paths, filters


@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD", auto_memory=True)
@makes_image(PIXINSIGHT, [POST_PROCESSED], target_filename,
             identity=TargetId, presentations=[THUMBNAIL])
def post_process_target(target: Target, mock_pixinsight: bool = False) -> None:
    print(f"Entering post_process_target for {target}")
    image_paths, image_filters = _target_channel_paths(target, PIXINSIGHT)
    images = Image.open_many(image_paths, filters=image_filters)

    result = _post_process_target(images, target)
    if result is None:
        return

    stars, starless = result
    if stars is not None:
        stars.save_to(target_filename(target, "stars"), mock=mock_pixinsight)
    if starless is not None:
        starless.save_to(target_filename(target, "starless"), mock=mock_pixinsight)


@node
@makes_image(OSS, [POST_PROCESSED], target_filename,
             identity=TargetId, presentations=[THUMBNAIL])
def post_process_target_oss(target: Target, mock_pixinsight: bool = False) -> None:
    """Stub: target post-processing for the OSS pipeline (TODO)."""
    return None
