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
from cache.context_ref import ContextRef
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


def _build_camera_fingerprint_map(
    metadata: MetadataDict,
    *,
    min_anchor_size: int,
) -> tuple[dict[FrameHash, tuple[int, ...]], list[FrameHash]]:
    """Cluster light frames by hot-pixel fingerprint within each
    (INSTRUME, NAXIS, XPIXSZ) bucket.

    Returns:
        fingerprint_map: per-frame canonical consensus tuple. The hash of
            this tuple becomes the camera identity for any Stack containing
            the frame. Same value across every member of the same physical
            sensor cluster.
        excluded: frames that lack a confident camera attribution (too few
            defects, or noise-fingerprint singletons). The caller drops
            these from stack identification.
    """
    by_pre_group: dict[tuple, list[FrameHash]] = {}
    for h, m in metadata.items():
        key = (
            m.header.get("INSTRUME"),
            m.header.get("NAXIS1"),
            m.header.get("NAXIS2"),
            m.header.get("XPIXSZ"),
            m.header.get("YPIXSZ"),
        )
        by_pre_group.setdefault(key, []).append(h)

    fingerprint_map: dict[FrameHash, tuple[int, ...]] = {}
    excluded: list[FrameHash] = []
    for hashes in by_pre_group.values():
        frame_sets = {h: metadata[h].defect_candidates for h in hashes}
        for cluster_frames, consensus in clustering.cluster_by_fingerprint(
            frame_sets, min_anchor_size=min_anchor_size
        ):
            if consensus:
                for h in cluster_frames:
                    fingerprint_map[h] = consensus
            else:
                excluded.extend(cluster_frames)
    return fingerprint_map, excluded


@node
def identify_stacks(
    metadata_ref: ContextRef[MetadataDict],
    angular_tolerance: float,
    light_frames: tuple[FrameHash, ...],
    min_frames_in_stack: int,
    max_run_gap,
    fingerprint_min_anchor_size: int,
) -> StacksDict:
    log.info("Identifying stacks", num_frames=len(light_frames))
    metadata_in = metadata_ref.get()

    # TODO: Maybe implement a version of cluster_continuous() that repeatedly
    # subdivides by independent 1D variables.
    metadata = MetadataDict(
        {h: i for h, i in metadata_in.items() if h in light_frames}
    )

    # Identify the physical camera that produced each light frame by
    # clustering on hot-pixel-defect overlap. Frames whose fingerprint
    # is absent or too sparse to attribute confidently are EXCLUDED
    # from stack identification entirely — we don't form Stacks from
    # frames we can't attribute to a camera. The bar is empirical:
    # ~0.12% on the validation library; if this jumps to several
    # percent, investigate (extract_defects regression or a new
    # camera/rig mix that the clusterer can't separate).
    fingerprint_map, excluded = _build_camera_fingerprint_map(
        metadata, min_anchor_size=fingerprint_min_anchor_size,
    )
    if excluded:
        excluded_meta = [metadata_in[h] for h in excluded]
        pct = 100 * len(excluded) / max(1, len(light_frames))
        (log.warning if pct < 5 else log.error)(
            "Light frames excluded from stack identification "
            "(no camera fingerprint)",
            count=len(excluded),
            total=len(light_frames),
            percent=round(pct, 2),
            sample_paths=[str(m.filename) for m in excluded_meta[:20]],
        )
    excluded_set = set(excluded)
    light_frames = tuple(h for h in light_frames if h not in excluded_set)
    metadata = MetadataDict(
        {h: i for h, i in metadata.items() if h not in excluded_set}
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
    ) -> Iterator[tuple[tuple[FrameHash, ...], tuple[int, ...]]]:
        # Two-level discrete clustering: first by header (INSTRUME, NAXIS,
        # XPIXSZ) — cheap and exact — then by camera fingerprint (looked
        # up from `fingerprint_map`, which was clustered once upfront).
        # Same-model bodies separate at the fingerprint step.
        cameras = [
            (
                metadata[k].content_hash,
                (
                    metadata[k].header["INSTRUME"],
                    metadata[k].header["NAXIS1"],
                    metadata[k].header["NAXIS2"],
                    metadata[k].header["XPIXSZ"],
                    metadata[k].header["YPIXSZ"],
                    fingerprint_map[k],
                ),
            )
            for k in frames
        ]
        for cluster in clustering.cluster_discrete(cameras):
            cluster_frames, keys = zip(*cluster)
            assert len(set(keys)) == 1
            camera_consensus = keys[0][-1]
            yield cluster_frames, camera_consensus

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
        for camera_frames, camera_consensus in camera_clusters(scale_frames):
            for rotation_frames in rotation_clusters(camera_frames):
                for filter_frames in filter_clusters(rotation_frames):
                    for exposure_frames in exposure_clusters(filter_frames):
                        if len(exposure_frames) >= min_frames_in_stack:
                            new_stack = Stack.from_frames(
                                exposure_frames,
                                metadata=metadata,
                                max_run_gap=max_run_gap,
                                camera_consensus=camera_consensus,
                            )
                            if new_stack is not None:
                                new_stacks[new_stack.stable_hash] = ContextRef.store(
                                    ("stacks", new_stack.stable_hash), new_stack
                                )

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
        stack = stack_ref.get()
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
