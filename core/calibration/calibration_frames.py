import ray
import numpy as np
import operator
import os
import itertools
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

from decorators.decorators import node
from metadata.classes import ImageMetadata, FrameHash
from stacks.types import StackHash
from stacks.run import calibration_run_from_frames
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
import clustering
from clustering.fingerprint import sensor_key
from clustering.quiesce import is_quiesced
from cache.context_ref import ContextRef
from metadata.classes import MetadataDict
from metadata.image_types import is_master
from beartype import beartype
import structlog

log = structlog.get_logger()


def _group_by_camera(
    metadata: MetadataDict,
    fingerprint_map,
) -> tuple[
    dict[tuple, set[FrameHash]],
    dict[tuple, tuple[int, ...]],
]:
    """Group calibration frames by physical camera identity.

    The key is `(sensor_key, consensus_tuple)` — geometry plus defect
    fingerprint, no header strings. Frames in `fingerprint_map` use the
    consensus from that map (built globally upstream for light frames).
    Calibration-only frames — biases, low-defect flats — fall back to
    `(sensor_key, ())`; the empty consensus tells downstream
    `CalibrationStack.from_run` to leave `camera_id_hash=None`.

    `INSTRUME` used to lead this key. Dropping it strictly reduces false
    SPLITS (a renamed driver no longer forks one sensor's darks into two
    stacks) and adds no false MERGE the old key did not already have: two
    same-model bodies shared one INSTRUME then and share one sensor_key
    now, so unfingerprintable biases were already pooled across them.

    Returns:
        frames_for_camera: camera_key -> set of frame hashes
        consensus_for_camera: camera_key -> consensus tuple
    """
    frames_for_camera: dict[tuple, set[FrameHash]] = {}
    consensus_for_camera: dict[tuple, tuple[int, ...]] = {}
    for h, m in metadata.items():
        consensus = fingerprint_map.get(FrameHash(h), ())
        key = (sensor_key(m.header), consensus)
        frames_for_camera.setdefault(key, set()).add(FrameHash(h))
        consensus_for_camera[key] = consensus
    return frames_for_camera, consensus_for_camera


@node
def identify_calibration_stacks(
    metadata_ref: ContextRef[MetadataDict],
    max_run_gap,
    min_frames_in_stack: int,
    fingerprint_map_ref,
    frontier: datetime | None = None,
    quiesce_gate=None,
) -> CalibrationStacksDict:
    log.info("Identifying calibration stacks")
    metadata = metadata_ref.get()
    fingerprint_map = fingerprint_map_ref.get()

    # Filter out any frames that look like already-stacked masters (by
    # filename, BITPIX, or PROGRAM signature). `find_image_files` excludes
    # the obvious `master*.fit` pattern, but external pre-stacked masters
    # with misleading filenames or IMAGETYP can still reach us; this is
    # the second-line defense.
    metadata = MetadataDict({
        h: m for h, m in metadata.items() if not is_master(m.header)
    })

    def get_runs(frames: set[FrameHash]) -> list[list[FrameHash]]:
        if not len(frames):
            return []

        items = [(h, metadata[h].header["DATE-OBS"]) for h in frames]
        result = clustering.cluster_continuous(
            items, epsilon=max_run_gap
        )
        # Run-quiesce gate: a calibration run still being captured (or
        # still downloading) stays out until the library frontier proves
        # it finished — a half-downloaded flat run must not mint a stack
        # (2026-09-05: a truncated morning flat run minted as "aborted").
        gate = quiesce_gate or timedelta(minutes=15)
        result = [
            r for r in result
            if is_quiesced(max(t for _, t in r), frontier, gate)
        ]
        # A run of fewer than min_frames_in_stack frames is not actually
        # stackable — typically a lone straggler frame, or an externally
        # pre-stacked master that slipped past `find_image_files` because
        # its filename or IMAGETYP didn't follow the master-naming
        # convention. Either way, treating it as a 1-frame "stack" just
        # produces a guaranteed-failing `stack_files` task and a missing-
        # master file for downstream lights to choke on. Drop it here.
        return [
            [x[0] for x in r] for r in result if len(r) >= min_frames_in_stack
        ]

    def group_by_header(
        images: list[ImageMetadata], key: str, default_value: Any
    ) -> dict[str, set[FrameHash]]:
        key_fn = lambda i: i.header.get(key, default_value) or default_value
        sorted_images = sorted(images, key=key_fn)
        return {
            k: set(FrameHash(i.content_hash) for i in g)
            for k, g in itertools.groupby(sorted_images, key=key_fn)
        }

    frames_for_image_type = group_by_header(metadata.values(), "IMAGETYP", "")

    # RULE (user decision 2026-09-05): all frames of one calibration stack
    # come from a SINGLE directory. Two rigs shooting flats simultaneously
    # land in different per-rig download folders, so this guarantees their
    # frames can never pool into one stack even when neither carries a
    # camera fingerprint (same sensor model = same geometry key). Light
    # stacks legitimately span directories; calibration runs never do —
    # a session's calibration frames share one astronomical-night folder.
    import os
    frames_for_dir: dict[str, set[FrameHash]] = {}
    for h, m in metadata.items():
        frames_for_dir.setdefault(
            os.path.dirname(m.filename), set()
        ).add(FrameHash(h))
    # `camera_key` here is the synthetic tuple `(sensor_key, consensus_tuple)`
    # where `consensus_tuple` is empty for biases / low-defect frames that
    # cannot be fingerprinted. Frames sharing the same camera_key are from
    # the same physical sensor (or from the same sensor GEOMETRY if no
    # fingerprint is available — biases need session-affinity attribution
    # to choose between same-model bodies, deferred).
    frames_for_camera, consensus_for_camera = _group_by_camera(
        metadata, fingerprint_map
    )
    frames_for_exposure = group_by_header(metadata.values(), "EXPOSURE", 0.0)
    frames_for_filter = group_by_header(metadata.values(), "FILTER", "")

    def new_stack(
        frames: list[FrameHash],
        camera_consensus: tuple[int, ...],
    ) -> CalibrationStack:
        # Only this run's frames, not the whole non-master library: that was
        # pickled into every calibration run task and hashed into its key.
        run = ray.get(calibration_run_from_frames.remote(
            frames, MetadataDict({h: metadata[h] for h in sorted(frames)})))
        assert run.image_type in ["bias", "dark", "flat"]
        return CalibrationStack.from_run(run, camera_consensus=camera_consensus)

    stacks = CalibrationStacksDict()

    def add_stack(frames: list[FrameHash], consensus: tuple[int, ...]) -> None:
        # One malformed run — e.g. a flat sequence captured with per-frame
        # auto-exposure drift (unsupported, decision 2026-09-05) — must not
        # block calibration identification for every other camera and
        # filter: the ice-flat O run did exactly that for a full day.
        try:
            stack = new_stack(frames, consensus)
        except Exception as e:
            log.error(
                "Skipping unstackable calibration run",
                n_frames=len(frames),
                first_frame=metadata[frames[0]].filename if frames else None,
                error=str(e).splitlines()[-1] if str(e) else repr(e),
            )
            return
        stacks[StackHash(stack.stable_hash)] = stack

    for camera_key, dir_frames in itertools.product(
        frames_for_camera, frames_for_dir.values()
    ):
        consensus = consensus_for_camera[camera_key]
        bias_frames = frames_for_image_type.get(
            "bias", set()
        ) & frames_for_camera.get(camera_key, set()) & dir_frames
        for run in get_runs(bias_frames):
            add_stack(run, consensus)

        for exposure in frames_for_exposure:
            dark_frames = (
                frames_for_image_type.get("dark", set())
                & frames_for_camera.get(camera_key, set())
                & frames_for_exposure.get(exposure, set())
                & dir_frames
            )
            for run in get_runs(dark_frames):
                add_stack(run, consensus)

        # Flats group by EXPOSURE too, exactly like darks: a filter shot
        # twice at different metered exposures (aborted-then-reshot sky
        # flats, or dusk vs dawn) is two distinct sequences, each
        # internally uniform. Without the exposure key an 8h run pooled
        # them into one non-uniform run that _aggregate_exposure rejected
        # wholesale — losing both (2026-09-05 FLT156 R: 15.0s x10 and
        # 13.8s x10, 8 min apart, both dropped). A truly drifting
        # per-frame sequence still fragments below min_frames (correct —
        # unsupported). EXPOSURE is discrete-keyed, so a locked sequence
        # (all frames one value) stays one run.
        for filter in frames_for_filter:
            for exposure in frames_for_exposure:
                flat_frames = (
                    frames_for_image_type.get("flat", set())
                    & frames_for_camera.get(camera_key, set())
                    & frames_for_filter.get(filter, set())
                    & frames_for_exposure.get(exposure, set())
                    & dir_frames
                )
                for run in get_runs(flat_frames):
                    add_stack(run, consensus)

    log.info("Calibration stacks identified", keys=list(stacks.keys()))
    return stacks


@beartype
def _midpoint_s(a: datetime, b: datetime) -> float:
    return (a.timestamp() + b.timestamp()) * 0.5


def focal_length_of(s) -> float | None:
    """FOCALLEN (mm) from the first frame header that carries one, or None.

    A flat only corrects the optical train it was shot through, and
    FOCALLEN is that train's signature in every ASIAir header. None
    (external pre-stacked masters, stripped headers) acts as a wildcard
    downstream — refusal is only for a PROVEN mismatch; refusing on
    unknown would strand runs the way requiring camera identity once did.
    A star flat carries its group's focal length as an attribute (it has no
    frames of its own).
    """
    if getattr(s, "focal_length", None) is not None:
        return float(s.focal_length)
    metadata = s.all_metadata() if hasattr(s, "all_metadata") else s.metadata
    for m in metadata.values():
        v = m.header.get("FOCALLEN")
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return None


def same_optical_train(a: float | None, b: float | None) -> bool:
    """FOCALLEN match within 2% — absorbs scope-profile rounding
    (350 vs 351) while cleanly separating real trains (351 vs 1234).
    Unknown on either side matches: refuse on proven difference only."""
    if a is None or b is None:
        return True
    return abs(a - b) <= 0.02 * max(a, b)


def sensor_of(s) -> tuple[int, int]:
    """The physical sensor bucket for calibration matching: pixel dimensions.

    Works on anything with ImagingAttributes — a light Run or a
    CalibrationStack — so both sides of a match derive the key the same way.
    """
    return (s.width_pixels, s.height_pixels)


def frame_dirs(metadata: MetadataDict) -> frozenset[str]:
    """The folders these frames live in, for directory affinity between
    lights and the flats shot beside them. Pure: no filesystem calls."""
    return frozenset(os.path.dirname(str(m.filename)) for m in metadata.values())


def _build_cal_index(cal_stacks: dict[StackHash, CalibrationStack]):
    """Index calibration stacks for chronological matching, bucketed by SENSOR
    GEOMETRY and tagged with camera identity.

    These buckets used to be keyed on `CalibrationStack.camera` — the
    INSTRUME string. That was a live correctness bug: a light run whose
    driver had renamed itself between sessions looked up a bucket that did
    not exist, got `None`, and was integrated with no master at all.

    Geometry is the widest bucket that is still physically valid — a master
    with different pixel dimensions cannot calibrate the frame under any
    interpretation. `camera_id_hash` (the defect fingerprint) then narrows
    within the bucket via `_prefer_same_camera`, so the strong signal still
    wins wherever it exists without being able to starve a run of masters
    where it does not.

    Each entry carries its own `camera_id_hash` so that narrowing needs no
    second pass over cal_stacks.
    """
    bias_by_sensor: dict[
        tuple, list[tuple[float, float, str | None, StackHash]]
    ] = {}
    dark_by_sensor: dict[
        tuple, list[tuple[float, float, float, str | None, StackHash]]
    ] = {}
    flat_by_sensor_filter: dict[
        tuple[tuple, str], list[tuple[float, str | None, float | None, StackHash, frozenset[str]]]
    ] = {}

    for sid, s in cal_stacks.items():
        if getattr(s, "kind", None) == "star":
            continue   # star flats are the user's option, never a default
        mid = _midpoint_s(s.oldest_date, s.newest_date)
        sensor = sensor_of(s)
        cam = s.camera_id_hash
        if s.image_type == "bias":
            bias_by_sensor.setdefault(sensor, []).append(
                (mid, s.mean_temperature, cam, sid)
            )
        elif s.image_type == "dark":
            dark_by_sensor.setdefault(sensor, []).append(
                (mid, s.mean_temperature, s.exposure, cam, sid)
            )
        elif s.image_type == "flat":
            flat_by_sensor_filter.setdefault((sensor, s.filter), []).append(
                (mid, cam, focal_length_of(s), sid, frame_dirs(s.all_metadata()))
            )

    # Sort each bucket by midpoint; helps cache locality, not required for
    # correctness (every matcher takes an explicit min over its candidates).
    for lst in bias_by_sensor.values():
        lst.sort(key=lambda x: x[0])
    for lst in dark_by_sensor.values():
        lst.sort(key=lambda x: x[0])
    for lst in flat_by_sensor_filter.values():
        lst.sort(key=lambda x: x[0])

    return bias_by_sensor, dark_by_sensor, flat_by_sensor_filter


@node
def associate_calibration_stacks(
    decision_overrides,
    cal_stacks: CalibrationStacksDict,
) -> CalibrationStacksDict:
    """Resolve cal-to-cal dependencies (e.g. a master bias for each flat)
    through the SAME decision framework that resolves light calibration:
    `resolve_calibration`, keyed on the shared `CALIBRATION_TOPOLOGY`.

    One mechanism, one topology — so cal-to-cal associations get user
    overrides and directory-affinity for free, and there is no second
    matcher/map to drift from the light path (that divergence is what left
    master darks bias-pre-calibrated against Landmann §6.2). Imported lazily
    to avoid a module-level cycle (resolve imports back into calibration).

    `resolve_calibration` returns `replace(stack, contents=...)`, so each
    `CalibrationStack`'s type and `camera_id_hash` survive unchanged.
    """
    from decisions.resolve import resolve_calibration
    return CalibrationStacksDict({
        k: resolve_calibration(cs, cal_stacks, decision_overrides)
        for k, cs in cal_stacks.items()
    })
