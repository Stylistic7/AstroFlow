import ray
import numpy as np
import operator
import itertools
from dataclasses import replace
from datetime import datetime
from typing import Any

from decorators.decorators import node
from metadata.classes import ImageMetadata, FrameHash
from stacks.types import StackHash
from stacks.run import calibration_run_from_frames
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict
import clustering
from clustering.fingerprint import sensor_key
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
        run = ray.get(calibration_run_from_frames.remote(frames, metadata))
        assert run.image_type in ["bias", "dark", "flat"]
        return CalibrationStack.from_run(run, camera_consensus=camera_consensus)

    stacks = CalibrationStacksDict()
    for camera_key in frames_for_camera:
        consensus = consensus_for_camera[camera_key]
        bias_frames = frames_for_image_type.get(
            "bias", set()
        ) & frames_for_camera.get(camera_key, set())
        for run in get_runs(bias_frames):
            stack = new_stack(run, consensus)
            stacks[StackHash(stack.stable_hash)] = stack

        for exposure in frames_for_exposure:
            dark_frames = (
                frames_for_image_type.get("dark", set())
                & frames_for_camera.get(camera_key, set())
                & frames_for_exposure.get(exposure, set())
            )
            for run in get_runs(dark_frames):
                stack = new_stack(run, consensus)
                stacks[StackHash(stack.stable_hash)] = stack

        for filter in frames_for_filter:
            flat_frames = (
                frames_for_image_type.get("flat", set())
                & frames_for_camera.get(camera_key, set())
                & frames_for_filter.get(filter, set())
            )
            for run in get_runs(flat_frames):
                stack = new_stack(run, consensus)
                stacks[StackHash(stack.stable_hash)] = stack

    log.info("Calibration stacks identified", keys=list(stacks.keys()))
    return stacks


@beartype
def _midpoint_s(a: datetime, b: datetime) -> float:
    return (a.timestamp() + b.timestamp()) * 0.5


def sensor_of(s) -> tuple[int, int]:
    """The physical sensor bucket for calibration matching: pixel dimensions.

    Works on anything with ImagingAttributes — a light Run or a
    CalibrationStack — so both sides of a match derive the key the same way.
    """
    return (s.width_pixels, s.height_pixels)


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
        tuple[tuple, str], list[tuple[float, str | None, StackHash]]
    ] = {}

    for sid, s in cal_stacks.items():
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
                (mid, cam, sid)
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
