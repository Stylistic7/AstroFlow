import ray
import numpy as np
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from context import context
from decorators import cached, makes_image, makes_video, measure_performance, create_thumbnail, create_video
from previews.operations import resize_array
from previews.flat_correction import create_flat_correction_thumbnails
from previews.calibration import calibrate_subframe
from file_io.types import FitsFilePath
from stacks.composite_stack import CompositeStack
from dependency_graph import Graph, TaskDescriptor
from previews.operations import autostretch
from customization.subframe_selection import get_light_frames_sorted_by_stars


@ray.remote
def scatter_stack_thumbnails() -> Graph:
  stacks = context.stacks
  print(f'### scatter_stack_thumbnails for {len(stacks)} stacks')

  tasks: list[TaskDescriptor] = list()

  for stack in context.stacks.values():
    tasks.append((stack_thumbnail, stack))
    tasks.append((stack_thumbnail_video, stack))
    tasks.append((create_flat_correction_thumbnails, stack))

  return Graph({gather_stack_thumbnails: tasks})


def get_best_light_frame(stack: CompositeStack) -> FitsFilePath:
  metadata = stack.all_metadata()
  light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
  return Path(metadata[light_frames_by_stars[0]].filename)


@makes_image(CompositeStack).as_subframe(get_best_light_frame)
@cached.remote
@measure_performance
@create_thumbnail
def stack_thumbnail(stack: CompositeStack) -> FitsFilePath:
  return get_best_light_frame(stack)


@makes_video(CompositeStack).as_subframe()
@cached.remote(resources={'fastest_storage': 0.001})
@measure_performance
@create_video
def stack_thumbnail_video(stack: CompositeStack) -> Iterable[np.ndarray]:
  # Sort all the light frames (no matter which run they're from) by the
  # number of stars. The better images should be in front.
  width = context.subframe_thumbnail_width
  metadata = stack.all_metadata()
  for light_frame in get_light_frames_sorted_by_stars(stack):
    light_frame = metadata[light_frame]
    dependencies = {}
    if substack := next(
        iter(s for s in stack.contents.values()
             if light_frame.content_hash in s.frames), None):
      dependencies = substack.dependencies

    im = calibrate_subframe(light_frame, dependencies, width)
    im = resize_array(im, width)
    im = autostretch(np.clip(im, 0.0, 1.0))
    im = (im * 255.0).astype(np.uint8)
    yield im


@ray.remote
def gather_stack_thumbnails() -> None:
  pass
