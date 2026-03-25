import cv2
import ray
import numpy as np
from beartype import beartype
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from context import context
from decorators import (
    node,
    create_thumbnail,
    create_video,
)
from previews.operations import resize_array
from previews.flat_correction import create_flat_correction_thumbnails
from frozendict import frozendict
from previews.calibration import calibrate_subframe
from file_io.types import FitsFilePath, WithSidecar
from stacks.simple_stack import Run, CalibrationStack
from dependency_graph import Graph, TaskDescriptor
from previews.operations import autostretch
from stacks.composite_stack import Stack
from customization.subframe_selection import get_light_frames_sorted_by_stars
from customization.resolve import is_frame_included
from products import makes_image, makes_video
from file_io.generic import read_image


@ray.remote
@beartype
def scatter_stack_thumbnails() -> Graph:
    tasks: list[TaskDescriptor] = list()

    calibration_stacks = context.calibration_stacks
    subframe_thumbnail_width = context.settings.preview_settings.subframe_thumbnail_width
    all_decisions = context.settings.decision_overrides

    def decisions_for(stack_hash):
        return frozendict({k: v for k, v in all_decisions.items() if len(k) >= 3 and k[2] == stack_hash})

    for stack_hash, stack_ref in context.stacks.items():
        stack = stack_ref.get()
        tasks.append((stack_thumbnail, stack))
        tasks.append((stack_thumbnail_video, stack, subframe_thumbnail_width, decisions_for(stack_hash)))
        tasks.append((create_flat_correction_thumbnails, stack, calibration_stacks, subframe_thumbnail_width))

    for stack in calibration_stacks.values():
        tasks.append((calibration_stack_thumbnail, stack))
        tasks.append((calibration_stack_thumbnail_video, stack, subframe_thumbnail_width))

    return Graph({gather_stack_thumbnails: tasks})


def get_first_frame(stack: Run) -> WithSidecar:
    # Use a sidecar file to avoid ever modifying the user's raw data files.
    return WithSidecar(
        FitsFilePath(stack.metadata[next(iter(stack.frames))].filename)
    )


def get_best_light_frame(stack: Stack) -> WithSidecar:
    metadata = stack.all_metadata()
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)

    # Use a sidecar file to avoid ever modifying the user's raw data files.
    return WithSidecar(
        FitsFilePath(metadata[light_frames_by_stars[0]].filename)
    )


@makes_image(Stack).as_subframe(get_best_light_frame)
@node
@create_thumbnail
def stack_thumbnail(stack: Stack) -> WithSidecar:
    return get_best_light_frame(stack)


@makes_image(CalibrationStack).as_subframe(get_first_frame)
@node
@create_thumbnail
def calibration_stack_thumbnail(stack: Run) -> WithSidecar:
    return get_first_frame(stack)


def _annotate_frame(
    im: np.ndarray, num_stars: int, included: bool,
) -> np.ndarray:
    """Burn star count and include/exclude status into a grayscale frame."""
    h, w = im.shape

    # Star count — top-left, white text with dark outline for readability.
    text = f"{num_stars}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 1200)
    thickness = max(1, int(scale * 2))
    pos = (int(w * 0.03), int(h * 0.07))
    cv2.putText(im, text, pos, font, scale, 0, thickness + 2, cv2.LINE_AA)
    cv2.putText(im, text, pos, font, scale, 255, thickness, cv2.LINE_AA)

    if not included:
        # Excluded frames: dim bar across the bottom + "EXCLUDED" label.
        bar_h = max(2, int(h * 0.02))
        im[h - bar_h:, :] = 200
        label_pos = (int(w * 0.03), h - int(h * 0.04))
        cv2.putText(im, "EXCLUDED", label_pos, font, scale * 0.8, 0, thickness + 2, cv2.LINE_AA)
        cv2.putText(im, "EXCLUDED", label_pos, font, scale * 0.8, 200, thickness, cv2.LINE_AA)

    return im


@makes_video(Stack).as_subframe()
@node(resources={"fastest_storage": 0.001})
@create_video
def stack_thumbnail_video(
    stack: Stack,
    subframe_thumbnail_width: int,
    decisions: frozendict = frozendict(),
) -> Iterable[np.ndarray]:
    # Sort all the light frames (no matter which run they're from) by the
    # number of stars. The better images should be in front.
    width = subframe_thumbnail_width
    metadata = stack.all_metadata()
    for frame_hash in get_light_frames_sorted_by_stars(stack):
        light_frame = metadata[frame_hash]
        dependencies = frozendict()
        if substack := next(
            iter(
                s
                for s in stack.contents.values()
                if light_frame.content_hash in s.frames
            ),
            None,
        ):
            dependencies = substack.dependencies

        im = calibrate_subframe(light_frame, dependencies, width)
        im = resize_array(im, width)
        im = autostretch(np.clip(im, 0.0, 1.0))
        im = (im * 255.0).astype(np.uint8)

        num_stars = light_frame.plate_solution.num_stars if light_frame.plate_solution else 0
        included = is_frame_included(frame_hash, stack, decisions)
        im = _annotate_frame(im, num_stars, included)

        yield im


@makes_video(CalibrationStack).as_subframe()
@node(resources={"fastest_storage": 0.001})
@create_video
def calibration_stack_thumbnail_video(
    stack: Run,
    subframe_thumbnail_width: int,
) -> Iterable[np.ndarray]:
    width = subframe_thumbnail_width
    for frame in stack.frames:
        im = autostretch(
            resize_array(read_image(stack.metadata[frame].filename), width)
        )
        yield (im * 255.0).astype(np.uint8)


@ray.remote
@beartype
def gather_stack_thumbnails() -> None:
    pass
