import ray
import numpy as np
import operator
import itertools
from dataclasses import replace
from datetime import datetime
from typing import Any
from frozendict import frozendict

from decorators.decorators import node
from metadata.classes import ImageMetadata, FrameHash
from stacks.types import StackHash, StackIdDependency
from stacks.run import BaseRun, Run, RunsDict, run_from_frames, calibration_run_from_frames
from stacks.composite_stack import Stack, CalibrationStack, CalibrationStacksDict
import clustering
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

    The key is `(INSTRUME, consensus_tuple)`. Frames in `fingerprint_map`
    use the consensus from that map (built globally upstream for light
    frames). Calibration-only frames — biases, low-defect flats — fall
    back to `(INSTRUME, ())`; the empty consensus tells downstream
    `CalibrationStack.from_run` to leave `camera_id_hash=None`.

    Returns:
        frames_for_camera: camera_key -> set of frame hashes
        consensus_for_camera: camera_key -> consensus tuple
    """
    frames_for_camera: dict[tuple, set[FrameHash]] = {}
    consensus_for_camera: dict[tuple, tuple[int, ...]] = {}
    for h, m in metadata.items():
        instr = m.header.get("INSTRUME") or ""
        consensus = fingerprint_map.get(FrameHash(h), ())
        key = (instr, consensus)
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
    # `camera_key` here is the synthetic tuple `(INSTRUME, consensus_tuple)`
    # where `consensus_tuple` is empty for biases / low-defect frames that
    # cannot be fingerprinted. Frames sharing the same camera_key are from
    # the same physical sensor (or from the same INSTRUME bucket if no
    # fingerprint is available — biases need session-affinity attribution
    # to choose between sensors of the same model, deferred).
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


def _build_cal_index(cal_stacks: dict[StackHash, CalibrationStack]):
    bias_by_cam: dict[str, list[tuple[float, float, StackHash]]] = {}
    dark_by_cam: dict[str, list[tuple[float, float, float, StackHash]]] = {}
    flat_by_cam_filter: dict[
        tuple[str, str], list[tuple[float, StackHash]]
    ] = {}

    for sid, s in cal_stacks.items():
        mid = _midpoint_s(s.oldest_date, s.newest_date)
        cam = s.camera
        if s.image_type == "bias":
            bias_by_cam.setdefault(cam, []).append(
                (mid, s.mean_temperature, sid)
            )
        elif s.image_type == "dark":
            dark_by_cam.setdefault(cam, []).append(
                (mid, s.mean_temperature, s.exposure, sid)
            )
        elif s.image_type == "flat":
            flat_by_cam_filter.setdefault((cam, s.filter), []).append(
                (mid, sid)
            )

    # Optionally sort each bucket by midpoint to help cache locality; not required for correctness.
    for lst in bias_by_cam.values():
        lst.sort(key=lambda x: x[0])
    for lst in dark_by_cam.values():
        lst.sort(key=lambda x: x[0])
    for lst in flat_by_cam_filter.values():
        lst.sort(key=lambda x: x[0])

    return bias_by_cam, dark_by_cam, flat_by_cam_filter


def associate_with_stacks(
    stacks: RunsDict, cal_stacks: CalibrationStacksDict
) -> RunsDict:
    bias_by_cam, dark_by_cam, flat_by_cam_filter = _build_cal_index(cal_stacks)

    def choose_bias(sub: BaseRun) -> StackHash | None:
        cand = bias_by_cam.get(sub.camera)
        if not cand:
            return None
        sub_mid = _midpoint_s(sub.oldest_date, sub.newest_date)
        sub_T = sub.mean_temperature
        cand = [t for (m, t, sid) in cand if abs(t - sub_T) < 10.0]
        if not cand:
            return None
        # Re-scan once to pick nearest by time; avoid NumPy
        best_sid = min(
            (
                (m, t, sid)
                for (m, t, sid) in bias_by_cam[sub.camera]
                if abs(t - sub_T) < 10.0
            ),
            key=lambda x: abs(x[0] - sub_mid),
        )[2]
        return best_sid

    def choose_dark(sub: BaseRun) -> StackHash | None:
        cand = dark_by_cam.get(sub.camera)
        if not cand:
            return None
        sub_mid = _midpoint_s(sub.oldest_date, sub.newest_date)
        sub_T = sub.mean_temperature
        sub_exp = sub.exposure
        filtered = [
            (m, t, e, sid)
            for (m, t, e, sid) in cand
            if abs(t - sub_T) < 10.0 and e <= sub_exp
        ]
        if not filtered:
            return None
        return min(filtered, key=lambda x: abs(x[0] - sub_mid))[3]

    def choose_flat(sub: BaseRun) -> StackHash | None:
        cand = flat_by_cam_filter.get((sub.camera, sub.filter))
        if not cand:
            return None
        sub_mid = _midpoint_s(sub.oldest_date, sub.newest_date)
        return min(cand, key=lambda x: abs(x[0] - sub_mid))[1]

    pickers = {
        "bias": choose_bias,
        "dark": choose_dark,
        "flat": choose_flat,
    }

    calibrations_for_stack_type = {
        "bias": [],
        "dark": ["bias"],
        "flat": ["bias"],
        "light": ["bias", "dark", "flat"],
    }

    new_stacks = RunsDict()
    for h, stack in stacks.items():
        deps = dict(stack.dependencies)  # mutable copy to build up
        for t in calibrations_for_stack_type[stack.image_type]:
            if t not in deps:
                sid = pickers[t](stack)

                if sid is not None:
                    deps[t] = StackIdDependency(stack_id=sid)
        new_stacks[h] = replace(stack, dependencies=frozendict(deps))

    return new_stacks


def associate_with_stack(
    stack: Stack, cal_stacks: CalibrationStacksDict
) -> Stack:
    new_contents = associate_with_stacks(stack.contents, cal_stacks)
    return replace(stack, contents=new_contents)


@node
def associate_calibration_stacks(
    cal_stacks: CalibrationStacksDict,
) -> CalibrationStacksDict:
    # Associate calibration stacks with each other (e.g., bias for darks/flats).
    # Extract runs, associate, then re-wrap.
    runs = RunsDict({k: next(iter(s.contents.values())) for k, s in cal_stacks.items()})
    associated = associate_with_stacks(runs, cal_stacks)
    return CalibrationStacksDict({
        k: CalibrationStack.from_run(associated[k])
        for k in cal_stacks
    })
