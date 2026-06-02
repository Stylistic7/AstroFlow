import math
import structlog
import ray
import numpy as np
from beartype import beartype
from frozendict import frozendict

from typing import Iterator
import clustering
from clustering.fingerprint import FingerprintMap
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


@node
def identify_camera_fingerprints(
    metadata_ref: ContextRef[MetadataDict],
    fingerprint_min_anchor_size: int,
) -> ContextRef[FingerprintMap]:
    """Cluster ALL light frames by hot-pixel fingerprint to assign each
    frame to a physical camera. Runs once, ABOVE the spatial-group split,
    so the clusterer sees the camera's full frame population — per-spatial-
    group clustering would never reach `min_anchor_size` for a camera
    spread across many pointings.

    Returns a `ContextRef[FingerprintMap]`. The map contains only frames
    whose attribution was confident; frames absent from the map (low-
    defect flats, noise fingerprints, or biases) are excluded downstream
    by `identify_stacks`. Wrapping in a ContextRef keeps downstream
    @node cache keys cheap to compute — 347+ identify_stacks tasks all
    consume the same ref, so we hash the FingerprintMap content once.
    """
    metadata = metadata_ref.get()
    log.info("Identifying camera fingerprints", num_frames=len(metadata))

    by_pre_group: dict[tuple, list[FrameHash]] = {}
    for h, m in metadata.items():
        if m.defect_candidates is None:
            continue
        key = (
            m.header.get("INSTRUME"),
            m.header.get("NAXIS1"),
            m.header.get("NAXIS2"),
            m.header.get("XPIXSZ"),
            m.header.get("YPIXSZ"),
        )
        by_pre_group.setdefault(key, []).append(h)

    fp = FingerprintMap()
    for hashes in by_pre_group.values():
        frame_sets = {h: metadata[h].defect_candidates for h in hashes}
        for cluster_frames, consensus in clustering.cluster_by_fingerprint(
            frame_sets, min_anchor_size=fingerprint_min_anchor_size,
        ):
            if not consensus:
                continue
            for h in cluster_frames:
                fp[h] = consensus
    log.info(
        "Camera fingerprints identified",
        attributed=len(fp),
        total=len(metadata),
    )
    return ContextRef.store_large(fp)


@node
def identify_stacks(
    metadata_ref: ContextRef[MetadataDict],
    angular_tolerance: float,
    light_frames: tuple[FrameHash, ...],
    min_frames_in_stack: int,
    max_run_gap,
    fingerprint_map_ref: ContextRef[FingerprintMap],
) -> StacksDict:
    log.info("Identifying stacks", num_frames=len(light_frames))
    metadata_in = metadata_ref.get()
    fingerprint_map = fingerprint_map_ref.get()

    # Frames whose fingerprint clustering didn't yield a confident
    # camera identity are EXCLUDED from stack identification entirely.
    # The fingerprint map was clustered globally upstream (see
    # `identify_camera_fingerprints`), so the count here reflects real
    # attribution failures, not a per-scale-group sample-size problem.
    # Empirically ~0.12% on the validation library; if this jumps to
    # several percent, investigate (extract_defects regression or a
    # new camera/rig mix that the clusterer can't separate).
    excluded = [h for h in light_frames if h not in fingerprint_map]
    if excluded:
        excluded_meta = [metadata_in[h] for h in excluded]
        pct = 100 * len(excluded) / max(1, len(light_frames))
        log.info(
            "Excluded frames (no camera fingerprint)",
            count=len(excluded),
            total=len(light_frames),
            percent=round(pct, 2),
            sample=str(excluded_meta[0].filename) if excluded_meta else None,
        )
    excluded_set = set(excluded)
    light_frames = tuple(h for h in light_frames if h not in excluded_set)
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
