import ray
from typing import Any
import os
from pathlib import Path
from astropy.time import Time

from context import context
from pix.decorators import PixInsight
from pix.classes import ImageList
from pix.tasks import (
    ChannelCombination,
    SpectrophotometricColorCalibration,
    ImageSolver,
    StarXTerminator,
    LinearFit,
    SaveImages,
    StarAlignment,
    LRGBCombination,
)
from file_io.types import XisfFilePath
from decorators import cached, measure_performance, create_thumbnail
from targets.types import Target
from stacks.artifacts import stack_filename
from products import makes_image


# TODO: Find a way to disable these during tests.
@PixInsight(slots=[1])
def _post_process_target(images: ImageList, target: Target) -> ImageList:
    print(f"### _post_process_target target {target} images {images}")

    if "LRGB" in images:
        aligned = StarAlignment(images.channels("LRGB"))
        rgb = LRGBCombination(aligned.channels("RGBL"))
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
        starless, stars = StarXTerminator(calibrated)

        output_filenames = [
            output_filename(target, "stars"),
            output_filename(target, "starless"),
        ]
        SaveImages([stars, starless], output_filenames)
    elif "RGB" in images:
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

        output_filenames = [
            output_filename(target, "stars"),
            output_filename(target, "starless"),
        ]
        SaveImages([stars, starless], output_filenames)

    if "SHO" in images:
        aligned = StarAlignment(images.channels("RGB"))
        _, s = StarXTerminator(aligned["S"])
        _, h = StarXTerminator(aligned["H"])
        _, o = StarXTerminator(aligned["O"])
        s, o = LinearFit([s, o], h)
        narrowband = ChannelCombination([s, h, o], "SHO")

        # TODO: Supply multiple output filenames.
        SaveImages([narrowband], [output_filename])


# TODO: Targets are composed of multiple color channels, so this needs to be
# able to return multiple images.
def target_filename(target: Target, suffix: str) -> XisfFilePath:
    return XisfFilePath(f"target {target.stable_hash}_{suffix}.xisf")


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@makes_image(Target).as_post_processed(target_filename)
@cached.remote(resources={"PixInsight": 1}, scheduling_strategy="SPREAD")
@measure_performance
@create_thumbnail
def post_process_target(target: Target) -> None:
    print(f"Entering post_process_target for {target}")
    stacks = context.stacks
    images = [stack_filename(stacks[stack_id]) for stack_id in target.stack_ids]

    result = _post_process_target(images, target)
    print(f"post_process_target, target {target} returning {result}")
    # return XisfFilePath(result)
