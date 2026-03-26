import tempfile
from pathlib import Path
from frozendict import frozendict

from previews.calibration import calibrate_subframe
from previews.operations import autostretch
from decorators import node, measure_performance
from stacks.composite_stack import Stack, RunsDict, CalibrationStacksDict
from customization.subframe_selection import get_light_frames_sorted_by_stars
from decorators import create_thumbnail
from stacks.run import Run
from stacks.types import StackHash, StackIdDependency
from file_io.tiff_file_io import write_tiff_image
from products import makes_image


# TODO: This should be a new, distinct type, FlatCorrectedStack.
@makes_image(Run).as_subframe()
@measure_performance
@create_thumbnail
def flat_corrected_stack(
    stack: Stack, substack: Run, flat_stack: Run, subframe_thumbnail_width: int
) -> tempfile.NamedTemporaryFile:
    width = subframe_thumbnail_width

    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    light_frame = stack.all_metadata()[
        next(l for l in light_frames_by_stars if l in substack.frames)
    ]

    dependencies = frozendict({**substack.dependencies, "flat": StackIdDependency(stack_id=StackHash(flat_stack.stable_hash))})

    temp_file = tempfile.NamedTemporaryFile(suffix=".tiff", delete=True)
    write_tiff_image(
        Path(temp_file.name),
        autostretch(calibrate_subframe(light_frame, dependencies, width)),
    )

    return temp_file


@node(memory=8 * 1024 * 1024 * 1024)
def create_flat_correction_thumbnails(stack: Stack, calibration_stacks: CalibrationStacksDict, subframe_thumbnail_width: int) -> None:
    flat_stack_ids = [
        id
        for id, s in calibration_stacks.items()
        if s.image_type == "flat"
        and s.camera == stack.camera
        and s.filter == stack.filter
    ]

    for substack in stack.contents.values():
        for flat_stack_id in flat_stack_ids:
            flat_corrected_stack(
                stack, substack, calibration_stacks[flat_stack_id], subframe_thumbnail_width
            )
