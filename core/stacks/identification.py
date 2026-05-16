import math
import structlog
import ray
import numpy as np
from beartype import beartype
from frozendict import frozendict

from typing import Iterator
import clustering
from stacks.composite_stack import Stack, StacksDict
from metadata.classes import FrameHash, MetadataDict
from cache.context_ref import ContextRef, deref
from context import context
from decorators import node
from decisions.decisions import (
    DecisionKey,
    DecisionPoint,
    DecisionType,
    BinaryChoice,
    ObjectType,
    resolve_decision,
)


log = structlog.get_logger()


@node
def identify_stacks(
    metadata_ref: ContextRef[MetadataDict],
    angular_tolerance: float,
    light_frames: tuple[FrameHash, ...],
    min_frames_in_stack: int,
    max_run_gap,
) -> StacksDict:
    log.info("Identifying stacks", num_frames=len(light_frames))
    metadata_in = metadata_ref.get()

    # TODO: Maybe implement a version of cluster_continuous() that repeatedly
    # subdivides by independent 1D variables.
    metadata = MetadataDict(
        {h: i for h, i in metadata_in.items() if h in light_frames}
    )

    @beartype
    def scale_clusters() -> Iterator[tuple[FrameHash, ...]]:
        scales = [
            (metadata[k].content_hash, metadata[k].plate_solution.image_scale)
            for k in metadata
        ]
        for cluster in clustering.cluster_continuous(scales, epsilon=0.2):
            scale_frames, _ = zip(*cluster)
            yield scale_frames

    @beartype
    def camera_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        cameras = [
            (
                metadata[k].content_hash,
                (
                    metadata[k].header["INSTRUME"],
                    metadata[k].header["NAXIS1"],
                    metadata[k].header["NAXIS2"],
                    metadata[k].header["XPIXSZ"],
                    metadata[k].header["YPIXSZ"],
                ),
            )
            for k in frames
        ]
        for cluster in clustering.cluster_discrete(cameras):
            frames, cameras = zip(*cluster)
            assert len(set(cameras)) == 1
            yield frames

    @beartype
    def rotation_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        rotation_angles = [
            (
                metadata[k].content_hash,
                metadata[k].plate_solution.rotation_angle,
            )
            for k in frames
        ]
        for cluster in clustering.cluster_angles_mod_pi(
            rotation_angles, angular_tolerance
        ):
            frames, _ = zip(*cluster)
            yield frames

    @beartype
    def filter_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        filters = [
            (metadata[k].content_hash, metadata[k].header["FILTER"])
            for k in frames
        ]
        for cluster in clustering.cluster_discrete(filters):
            filter_frames, filters = zip(*cluster)
            assert len(set(filters)) == 1
            yield filter_frames

    @beartype
    def exposure_clusters(
        frames: tuple[FrameHash, ...],
    ) -> Iterator[tuple[FrameHash, ...]]:
        # Continuous (not discrete) so 299.99s and 300.0s — capture-software
        # rounding jitter — group together. WBPP groups by EXPOSURE and emits
        # one master per group, so this is what makes AstroFlow's
        # one-stack-one-master assumption hold.
        exposures = [
            (metadata[k].content_hash, metadata[k].header["EXPOSURE"])
            for k in frames
        ]
        for cluster in clustering.cluster_continuous(exposures, epsilon=0.5):
            exposure_frames, _ = zip(*cluster)
            yield exposure_frames

    new_stacks = StacksDict()
    for scale_frames in scale_clusters():
        for camera_frames in camera_clusters(scale_frames):
            for rotation_frames in rotation_clusters(camera_frames):
                for filter_frames in filter_clusters(rotation_frames):
                    for exposure_frames in exposure_clusters(filter_frames):
                        if len(exposure_frames) >= min_frames_in_stack:
                            new_stack = Stack.from_frames(
                                exposure_frames, metadata=metadata, max_run_gap=max_run_gap
                            )
                            if new_stack is not None:
                                # Pure: return the real Stack. store_stacks
                                # (always-runs, uncached) does the
                                # ContextRef.store so the cached @node
                                # result is self-sufficient and can't
                                # point at a missing payload.
                                new_stacks[new_stack.stable_hash] = new_stack

    assert None not in new_stacks.values()

    # Detect and publish decision points for runs with notable rotation spread.
    # resolve_decision handles publishing to context.
    detected_decisions = _detect_rotation_outliers(
        new_stacks, metadata, angular_tolerance
    )
    for decision in detected_decisions.values():
        resolve_decision(decision, frozendict(), default=BinaryChoice.INCLUDE)

    log.info("Identified stacks", num_stacks=len(new_stacks))
    return new_stacks


def _detect_rotation_outliers(
    stacks: StacksDict,
    metadata: MetadataDict,
    angular_tolerance: float,
) -> dict:
    """Detect runs within stacks whose rotation angle is far from the stack mean.

    Flags any run where the per-run mean rotation angle deviates from the
    stack-wide mean by more than half the angular tolerance.
    """
    detected = {}

    for stack_hash, stack_ref in stacks.items():
        stack = deref(stack_ref)
        if len(stack.contents) < 2:
            continue

        # Compute per-run mean rotation angles.
        run_angles = {}
        for run_hash, run in stack.contents.items():
            angles = [
                m.plate_solution.rotation_angle
                for m in run.metadata.values()
                if m.plate_solution is not None
            ]
            if angles:
                run_angles[run_hash] = clustering.angular_mean_mod_pi(angles)

        if len(run_angles) < 2:
            continue

        spread = clustering.angular_spread_mod_pi(list(run_angles.values()))
        if spread <= angular_tolerance * 0.5:
            continue

        stack_mean = clustering.angular_mean_mod_pi(list(run_angles.values()))

        for run_hash, run_angle in run_angles.items():
            # Distance from mean, accounting for mod-pi periodicity.
            diff = clustering.angular_spread_mod_pi([run_angle, stack_mean])
            if diff <= angular_tolerance * 0.3:
                continue

            dp = DecisionPoint.make(
                key=DecisionKey.rotation_outlier(stack_hash, run_hash),
                title=f"Run rotation differs by {math.degrees(diff):.1f}°",
                description=(
                    f"This run's mean rotation angle differs by "
                    f"{math.degrees(diff):.1f}° from the stack mean. "
                    f"The stack's total rotation spread is "
                    f"{math.degrees(spread):.1f}° "
                    f"(tolerance: {math.degrees(angular_tolerance):.1f}°). "
                    f"Including it may cause alignment artifacts."
                ),
                metadata={
                    "run_angle_deg": round(math.degrees(run_angle), 2),
                    "stack_mean_deg": round(math.degrees(stack_mean), 2),
                    "diff_deg": round(math.degrees(diff), 2),
                    "spread_deg": round(math.degrees(spread), 2),
                    "tolerance_deg": round(math.degrees(angular_tolerance), 2),
                    "run_frame_count": len(stack.contents[run_hash].frames),
                },
            )
            detected[dp.key] = dp

    return detected
