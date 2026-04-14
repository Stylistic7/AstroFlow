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
from file_io.types import XisfFilePath
from decorators import node, create_thumbnail
from targets.types import Target
from stacks.artifacts import stack_filename
from stacks.composite_stack import StacksDict
from products import makes_image
from previews.types import QualityLevel, PIXINSIGHT, OSS


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


@node(resources={"PixInsight": 1}, scheduling_strategy="SPREAD")
@makes_image(Target, PIXINSIGHT, [QualityLevel.POST_PROCESSED], target_filename)
@create_thumbnail
def post_process_target(target: Target, stacks: StacksDict, mock_pixinsight: bool = False) -> None:
    print(f"Entering post_process_target for {target}")
    image_paths = [stack_filename(stacks[stack_id].get()) for stack_id in target.stack_ids]
    images = Image.open_many(image_paths)

    result = _post_process_target(images, target)
    if result is None:
        return

    stars, starless = result
    if stars is not None:
        stars.save_to(target_filename(target, "stars"), mock=mock_pixinsight)
    if starless is not None:
        starless.save_to(target_filename(target, "starless"), mock=mock_pixinsight)


@node
@makes_image(Target, OSS, [QualityLevel.POST_PROCESSED], target_filename)
@create_thumbnail
def post_process_target_oss(target: Target, stacks: StacksDict, mock_pixinsight: bool = False) -> None:
    """Stub: target post-processing for the OSS pipeline (TODO)."""
    return None
