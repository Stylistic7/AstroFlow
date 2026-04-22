import numpy as np
from frozendict import frozendict

from previews.calibration import calibrate_subframe
from previews.operations import autostretch
from decorators import node, measure_performance
from stacks.composite_stack import Stack, CalibrationStacksDict
from decisions.subframe_selection import get_light_frames_sorted_by_stars
from stacks.run import Run
from stacks.types import StackHash, StackIdDependency
from products import makes_image
from products.identity import PresentationType
from previews.identity import CalibratedStackId, UncorrectedRunId
from previews.types import ALL_IMPLEMENTATIONS, SUBFRAME

THUMBNAIL = PresentationType.THUMBNAIL


@makes_image(Run, ALL_IMPLEMENTATIONS, [SUBFRAME],
             identity=CalibratedStackId,
             presentations=[THUMBNAIL])
@measure_performance
def flat_corrected_stack(
    stack: Stack, substack: Run, flat_stack: Run,
) -> np.ndarray:
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_in_run = next(l for l in light_frames_by_stars if l in substack.frames)
    light_frame = stack.all_metadata()[best_in_run]

    dependencies = frozendict(
        {
            **substack.dependencies,
            "flat": StackIdDependency(
                stack_id=StackHash(flat_stack.stable_hash)
            ),
        }
    )

    return calibrate_subframe(light_frame, dependencies)


@makes_image(Run, ALL_IMPLEMENTATIONS, [SUBFRAME],
             identity=UncorrectedRunId,
             presentations=[THUMBNAIL])
def uncorrected_run_thumbnail(
    stack: Stack, substack: Run,
) -> np.ndarray:
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_in_run = next(l for l in light_frames_by_stars if l in substack.frames)
    light_frame = stack.all_metadata()[best_in_run]

    return calibrate_subframe(light_frame, substack.dependencies)


@node
def create_flat_correction_thumbnails(
    stack: Stack,
    calibration_stacks: CalibrationStacksDict,
    subframe_thumbnail_width: int,
) -> None:
    flat_stack_ids = [
        id
        for id, s in calibration_stacks.items()
        if s.image_type == "flat"
        and s.camera == stack.camera
        and s.filter == stack.filter
    ]

    for substack in stack.contents.values():
        uncorrected_run_thumbnail(stack, substack)
        for flat_stack_id in flat_stack_ids:
            flat_corrected_stack(
                stack,
                substack,
                calibration_stacks[flat_stack_id],
            )
