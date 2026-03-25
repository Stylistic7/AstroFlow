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
from stacks.simple_stack import Run, RunsDict, run_from_frames
from stacks.composite_stack import Stack, CalibrationStack, CalibrationStacksDict
import clustering
from cache.context_ref import ContextRef
from metadata.classes import MetadataDict
from beartype import beartype
import structlog

log = structlog.get_logger()


@node
def identify_calibration_stacks(metadata_ref: ContextRef[MetadataDict], max_run_gap) -> RunsDict:
    log.info("Identifying calibration stacks")
    metadata = metadata_ref.get()

    def get_runs(frames: set[FrameHash]) -> list[list[FrameHash]]:
        if not len(frames):
            return []

        items = [(h, metadata[h].header["DATE-OBS"]) for h in frames]
        result = clustering.cluster_continuous(
            items, epsilon=max_run_gap
        )
        return [[x[0] for x in r] for r in result]

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
    frames_for_camera = group_by_header(metadata.values(), "INSTRUME", "")
    frames_for_exposure = group_by_header(metadata.values(), "EXPOSURE", 0.0)
    frames_for_filter = group_by_header(metadata.values(), "FILTER", "")

    def new_stack(frames: list[FrameHash]) -> CalibrationStack:
        run = ray.get(run_from_frames.remote(frames, metadata))
        assert run.image_type in ["bias", "dark", "flat"]
        return CalibrationStack.from_run(run)

    stacks = CalibrationStacksDict()
    for camera in frames_for_camera:
        bias_frames = frames_for_image_type.get(
            "bias", set()
        ) & frames_for_camera.get(camera, set())
        for run in get_runs(bias_frames):
            stack = new_stack(run)
            stacks[StackHash(stack.stable_hash)] = stack

        for exposure in frames_for_exposure:
            dark_frames = (
                frames_for_image_type.get("dark", set())
                & frames_for_camera.get(camera, set())
                & frames_for_exposure.get(exposure, set())
            )
            for run in get_runs(dark_frames):
                stack = new_stack(run)
                stacks[StackHash(stack.stable_hash)] = stack

        for filter in frames_for_filter:
            flat_frames = (
                frames_for_image_type.get("flat", set())
                & frames_for_camera.get(camera, set())
                & frames_for_filter.get(filter, set())
            )
            for run in get_runs(flat_frames):
                stack = new_stack(run)
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

    def choose_bias(sub: Run) -> StackHash | None:
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

    def choose_dark(sub: Run) -> StackHash | None:
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

    def choose_flat(sub: Run) -> StackHash | None:
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
        log.info("Associating stack", stack=str(stack))
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
