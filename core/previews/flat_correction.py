import numpy as np
from frozendict import frozendict

from previews.calibration import calibrate_subframe
from decorators import measure_performance
from stacks.composite_stack import Stack
from decisions.subframe_selection import get_light_frames_sorted_by_stars
from stacks.run import Run
from stacks.types import StackHash, StackIdDependency
from products.registry import image_producer


@image_producer()
@measure_performance
def calibrated_run_preview(
    stack: Stack,
    substack: Run,
    flat_id: str | None,
    dark_id: str | None,
    bias_id: str | None,
    fringe_id: str | None,
) -> np.ndarray:
    """One run's best subframe, calibrated with EXACTLY the stacks named here.

    Every calibration input is an explicit argument. Reading them off
    `substack.dependencies` is what broke this before: the route handed in an
    UNRESOLVED stack, whose run dependencies are always empty, so the flat
    divided a frame that still carried its ~87% bias+dark pedestal. That
    stamps the flat's own pattern onto the pedestal at several times the
    signal amplitude, which is the washed-out preview (2026-09-08).

    Naming them also puts them in the cache key, so two previews that differ
    only in their dark are two products rather than one stale one. `None`
    means "do not apply this kind", which is how the uncorrected view is
    spelled — one producer covers both, so they can never disagree.

    `fringe_id` is the fourth axis, named like the rest so that a grid cell
    showing (flat A, fringe B) is a DIFFERENT cached product from (flat A,
    no fringe) — which is the whole point of a comparison grid.
    """
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
    best_in_run = next(l for l in light_frames_by_stars if l in substack.frames)
    light_frame = stack.all_metadata()[best_in_run]

    deps = {
        kind: StackIdDependency(stack_id=StackHash(cal_id))
        for kind, cal_id in (
            ("flat", flat_id), ("dark", dark_id), ("bias", bias_id),
            ("fringe", fringe_id),
        )
        if cal_id is not None
    }
    return calibrate_subframe(light_frame, frozendict(deps))
