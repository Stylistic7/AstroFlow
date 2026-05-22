import numpy as np
from frozendict import frozendict

from previews.calibration import calibrate_subframe
from previews.operations import autostretch
from decorators import node, measure_performance
from stacks.composite_stack import Stack, CalibrationStack
from decisions.subframe_selection import get_light_frames_sorted_by_stars
from stacks.run import Run
from stacks.types import StackHash, StackIdDependency
from products import makes_image
from products.identity import PresentationType
from previews.identity import CalibratedStackId, UncorrectedRunId

THUMBNAIL = PresentationType.THUMBNAIL

@makes_image(identity=CalibratedStackId)
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

@makes_image(identity=UncorrectedRunId)
def uncorrected_run_thumbnail(
    stack: Stack, substack: Run,
) -> np.ndarray:
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_in_run = next(l for l in light_frames_by_stars if l in substack.frames)
    light_frame = stack.all_metadata()[best_in_run]

    # `substack.dependencies` has an auto-picked flat from resolve_calibration.
    # Strip it so this thumbnail is genuinely uncorrected — otherwise it looks
    # identical to the default "corrected" option in the UI.
    dependencies = frozendict(
        {k: v for k, v in substack.dependencies.items() if k != "flat"}
    )
    return calibrate_subframe(light_frame, dependencies)

@node
def create_flat_correction_thumbnails(
    stack: Stack,
    matching_flat_stacks: tuple[CalibrationStack, ...],
) -> None:
    """Render an "uncorrected" thumbnail plus one per matching flat for each
    substack of `stack`.

    `stack` is expected to already be calibration-resolved (bias / dark
    populated on each substack's dependencies). `matching_flat_stacks` is
    pre-filtered at the scatter level to only the flats that match this
    stack's camera + filter. Narrow inputs let the cache key invalidate
    only when something this function actually uses changes — not on
    every unrelated calibration-stack add elsewhere in the catalog.
    """
    for substack in stack.contents.values():
        uncorrected_run_thumbnail(stack, substack)
        for flat_stack in matching_flat_stacks:
            flat_corrected_stack(stack, substack, flat_stack)
