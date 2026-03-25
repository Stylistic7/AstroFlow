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
from stacks.simple_stack import SimpleStack, CalibrationStack
from dependency_graph import Graph, TaskDescriptor
from previews.operations import autostretch
from stacks.composite_stack import CompositeStack
from customization.subframe_selection import get_light_frames_sorted_by_stars
from products import makes_image, makes_video
from file_io.generic import read_image


@ray.remote
@beartype
def scatter_stack_thumbnails() -> Graph:
    tasks: list[TaskDescriptor] = list()

    calibration_stacks = context.calibration_stacks
    subframe_thumbnail_width = context.settings.preview_settings.subframe_thumbnail_width

    for stack_ref in context.stacks.values():
        stack = stack_ref.get()
        tasks.append((stack_thumbnail, stack))
        tasks.append((stack_thumbnail_video, stack, subframe_thumbnail_width))
        tasks.append((create_flat_correction_thumbnails, stack, calibration_stacks, subframe_thumbnail_width))

    for stack in calibration_stacks.values():
        tasks.append((calibration_stack_thumbnail, stack))
        tasks.append((calibration_stack_thumbnail_video, stack, subframe_thumbnail_width))

    return Graph({gather_stack_thumbnails: tasks})


def get_first_frame(stack: SimpleStack) -> WithSidecar:
    # Use a sidecar file to avoid ever modifying the user's raw data files.
    return WithSidecar(
        FitsFilePath(stack.metadata[next(iter(stack.frames))].filename)
    )


def get_best_light_frame(stack: CompositeStack) -> WithSidecar:
    metadata = stack.all_metadata()
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)

    # Use a sidecar file to avoid ever modifying the user's raw data files.
    return WithSidecar(
        FitsFilePath(metadata[light_frames_by_stars[0]].filename)
    )


@makes_image(CompositeStack).as_subframe(get_best_light_frame)
@node
@create_thumbnail
def stack_thumbnail(stack: CompositeStack) -> WithSidecar:
    return get_best_light_frame(stack)


@makes_image(CalibrationStack).as_subframe(get_first_frame)
@node
@create_thumbnail
def calibration_stack_thumbnail(stack: SimpleStack) -> WithSidecar:
    return get_first_frame(stack)


@makes_video(CompositeStack).as_subframe()
@node(resources={"fastest_storage": 0.001})
@create_video
def stack_thumbnail_video(stack: CompositeStack, subframe_thumbnail_width: int) -> Iterable[np.ndarray]:
    # Sort all the light frames (no matter which run they're from) by the
    # number of stars. The better images should be in front.
    width = subframe_thumbnail_width
    metadata = stack.all_metadata()
    for light_frame in get_light_frames_sorted_by_stars(stack):
        light_frame = metadata[light_frame]
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
        yield im


@makes_video(CalibrationStack).as_subframe()
@node(resources={"fastest_storage": 0.001})
@create_video
def calibration_stack_thumbnail_video(
    stack: SimpleStack,
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
