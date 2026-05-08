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
    create_video,
)
from previews.operations import resize_array
from previews.flat_correction import create_flat_correction_thumbnails
from frozendict import frozendict
from previews.calibration import calibrate_subframe
from file_io.types import FitsFilePath, WithSidecar
from stacks.run import Run
from stacks.composite_stack import CalibrationStack, CalibrationStacksDict, StacksDict
from cache.context_ref import ContextRef
from dependency_graph import Graph, TaskDescriptor
from previews.operations import autostretch
from stacks.composite_stack import Stack
from decisions.subframe_selection import get_light_frames_sorted_by_stars
from decisions.resolve import is_frame_included, resolve_calibration
from products import makes_image, makes_video
from products.identity import StackId, PresentationType
from previews.types import ALL_IMPLEMENTATIONS, SUBFRAME

THUMBNAIL = PresentationType.THUMBNAIL
from file_io.generic import read_image
from decisions.settings import ProcessingConfig
from decisions.decisions import decisions_for_stack


@ray.remote(num_cpus=0.1)
@beartype
def scatter_stack_thumbnails(
    cal_stacks_ref: ContextRef,
    stacks: StacksDict,
) -> Graph:
    """Build the per-stack-thumbnail subgraph.

    Receives `cal_stacks_ref` and `stacks` from the parent
    `build_preview_graph` rather than reading them from the
    ContextActor directly. The dep-graph already orders this task
    after regroup/store completes — passing the data through enforces
    that ordering at the type level rather than relying on the
    ContextActor having been written to in time.
    """
    tasks: list[TaskDescriptor] = list()

    settings = context.settings
    subframe_thumbnail_width = settings.preview_settings.subframe_thumbnail_width

    def config_for(stack_hash):
        decisions = decisions_for_stack(settings.decision_overrides, stack_hash)
        return ProcessingConfig(
            calibration_stacks=cal_stacks_ref,
            development_settings=settings.development_settings,
            min_frames_in_stack=settings.clustering_settings.min_frames_in_stack,
            decisions=decisions,
        )

    calibration_stacks = cal_stacks_ref.get() or {}
    for stack_hash, stack_ref in stacks.items():
        stack = stack_ref.get()
        tasks.append((stack_thumbnail, stack))
        tasks.append((stack_thumbnail_video, stack, subframe_thumbnail_width, config_for(stack_hash)))

        # Pre-narrow inputs to create_flat_correction_thumbnails so its cache
        # key invalidates only on changes that actually affect its output.
        # Passing the full calibration_stacks made every new bias/dark/flat
        # added anywhere in the catalog miss the cache for every light stack.
        decisions = decisions_for_stack(settings.decision_overrides, stack_hash)
        resolved_stack = resolve_calibration(stack, calibration_stacks, decisions)
        matching_flat_stacks = tuple(
            s for s in calibration_stacks.values()
            if s.image_type == "flat"
            and s.camera == stack.camera
            and s.filter == stack.filter
        )
        tasks.append((create_flat_correction_thumbnails, resolved_stack, matching_flat_stacks))

    for stack in calibration_stacks.values():
        tasks.append((calibration_stack_thumbnail, stack))
        tasks.append((calibration_stack_thumbnail_video, stack, subframe_thumbnail_width))

    return Graph({gather_stack_thumbnails: tasks})


def get_first_frame(stack: CalibrationStack) -> WithSidecar:
    # Use a sidecar file to avoid ever modifying the user's raw data files.
    metadata = stack.all_metadata()
    first_frame = stack.all_frames()[0]
    return WithSidecar(FitsFilePath(metadata[first_frame].filename))


def get_best_light_frame(stack: Stack) -> WithSidecar:
    metadata = stack.all_metadata()
    light_frames_by_stars = get_light_frames_sorted_by_stars(stack)

    # Use a sidecar file to avoid ever modifying the user's raw data files.
    return WithSidecar(
        FitsFilePath(metadata[light_frames_by_stars[0]].filename)
    )


@node
@makes_image(Stack, ALL_IMPLEMENTATIONS, [SUBFRAME], get_best_light_frame,
             identity=StackId, presentations=[THUMBNAIL])
def stack_thumbnail(stack: Stack) -> WithSidecar:
    return get_best_light_frame(stack)


@node
@makes_image(CalibrationStack, ALL_IMPLEMENTATIONS, [SUBFRAME], get_first_frame,
             identity=StackId, presentations=[THUMBNAIL])
def calibration_stack_thumbnail(stack: CalibrationStack) -> WithSidecar:
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


def _stack_thumbnail_video_narrow(args, kwargs):
    """Cache-key hook: drop irrelevant cal_stacks and other-stack
    decisions from `config` for hashing. Function still receives the
    full ProcessingConfig at runtime — see `_stack_files_narrow` in
    `stacks/stacking.py` for the full rationale.
    """
    from stacks.stacking import narrow_stack_config
    stack, width, config, *rest = args
    return (stack, width, narrow_stack_config(stack, config), *rest), kwargs


@node(resources={"fastest_storage": 0.001},
      cache_key_narrow=_stack_thumbnail_video_narrow)
@makes_video(Stack, ALL_IMPLEMENTATIONS, [SUBFRAME])
@create_video
def stack_thumbnail_video(
    stack: Stack,
    subframe_thumbnail_width: int,
    config: ProcessingConfig,
) -> Iterable[np.ndarray]:
    # Resolve calibration lazily (for flat correction of subframes).
    stack = resolve_calibration(stack, config.get_calibration_stacks(), config.decisions)
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
        included = is_frame_included(frame_hash, stack, config.decisions)
        im = _annotate_frame(im, num_stars, included)

        yield im


@node(resources={"fastest_storage": 0.001})
@makes_video(CalibrationStack, ALL_IMPLEMENTATIONS, [SUBFRAME])
@create_video
def calibration_stack_thumbnail_video(
    stack: CalibrationStack,
    subframe_thumbnail_width: int,
) -> Iterable[np.ndarray]:
    width = subframe_thumbnail_width
    metadata = stack.all_metadata()
    for frame in stack.all_frames():
        im = autostretch(
            resize_array(read_image(metadata[frame].filename), width)
        )
        yield (im * 255.0).astype(np.uint8)


@ray.remote(num_cpus=0.1)
@beartype
def gather_stack_thumbnails() -> None:
    pass
