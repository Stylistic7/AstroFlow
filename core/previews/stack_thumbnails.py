import numpy as np
import cv2
import os
import tempfile
import ray
from typing import cast
from ray.remote_function import RemoteFunction

from dependency_graph.astroflow_node import AstroFlowNode
from stacks.types import StackHash
from metadata.classes import ImageMetadata
from previews.operations import resize_array, array_to_png
from previews.flat_correction import create_flat_correction_thumbnails
from previews.calibration import calibrate_subframe
from previews.types import QualityLevel
from context import context, WriteMode
from file_io.generic import read_image
# TODO: Unify quarantining
import quarantining
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from dependency_graph import Graph, TaskDescriptor
from previews.operations import autostretch
from profiling import track_performance

# TODO: Split this file up into smaller, more specific pieces.

# Every stack is going to have multiple levels of previews. I should disentangle
# them. There will always be a subrame-based preview. The master-based preview
# will only be computed after stacking.
#
# The subframe-based preview requires calibration inside AstroFlow, but
# the master-based preview is just a thumbnail of the master image.
#
# There will also be post-processed previews. It would be good for the user
# to scroll through all the different types.
#
# The keys in the cache should be distinct:
#
# ('stack_subframe_thumbnail_image', 'abc123') = image
# ('stack_master_thumbnail_image', 'abc123') = image
# ('stack_postprocessed_thumbnail_image', 'abc123') = image
#
# Better:
#
# ('stack_thumbnail_image', 'subframe', 'abc123') = image
# ('stack_thumbnail_image', 'master', 'abc123') = image
# ('stack_thumbnail_image', 'post_processed', 'abc123') = image


# TODO: This needs to use the Artifact workflow, so that it will automatically
# re-generate previews when stack contents change.
@AstroFlowNode(resources={'fastest_storage': 0.001})
def scatter_stack_thumbnails(mode: QualityLevel) -> Graph:
  stacks = context.stacks
  print(f'### scatter_stack_thumbnails for {len(stacks)} stacks')

  tasks: list[TaskDescriptor] = list()

  if mode == 'subframe':
    for stack in context.stacks.values():
      tasks.append((create_stack_subframe_thumbnails, stack))
      tasks.append((create_flat_correction_thumbnails, stack))

    # tasks.extend([(create_flat_correction_thumbnails, id)
    #               for id, s in stacks.items()
    #               if isinstance(s, SimpleStack) and s.image_type == 'light'])
    # tasks.extend([
    #     (create_stack_subframe_thumbnails, id)
    #     for id, s in stacks.items()
    #     if isinstance(s, CompositeStack) and s.image_type == 'light' and
    #     (not context.contains(('stack_subframe_thumbnail_image', id)) or
    #      not context.contains(('stack_subframe_thumbnail_video', id)))
    # ])
  elif mode == 'master':
    # TODO: Implement
    pass

    # tasks = [(create_stack_master_thumbnail, id)
    #          for id, s in stacks.items()
    #          if type(s) is CompositeStack and s.image_type == 'light' and
    #          ('stack_master_thumbnail_image', id) not in cache_keys]

  return Graph({cast(RemoteFunction, gather): tasks})


# TODO: Cache this.
@AstroFlowNode()
def create_stack_subframe_thumbnails(stack: CompositeStack) -> None:
  print(f'Entering create_stack_subframe_thumbnails() for stack {stack}')

  width = context.subframe_thumbnail_width

  # Now sort all the light frames (no matter which run they're from) by the
  # number of stars. The better images should be in front.
  light_frames_by_stars = quarantining.get_light_frames_sorted_by_stars(
      stack.all_frames())
  best_light_frame = light_frames_by_stars[0]
  image = resize_array(read_image(best_light_frame.filename), width)
  height, width = image.shape

  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(light_frames_by_stars)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # type: ignore
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)
    for i, light_frame in enumerate(light_frames_by_stars):
      dependencies = {}
      if substack := next(
          iter(s for s in stack.contents.values()
               if light_frame.content_hash in s.frames), None):
        dependencies = substack.dependencies

      im = calibrate_subframe(light_frame, dependencies, width)
      im = autostretch(np.clip(im, 0.0, 1.0))

      if i == 0:
        key = ('stack_subframe_thumbnail_image', stack.stable_hash)
        context.set(key, im, write_mode=WriteMode.DISK_ONLY)
        print(f'### Wrote key {key}')

        key = ('stack_subframe_thumbnail_image_png', stack.stable_hash)
        context.set(key, array_to_png(im), write_mode=WriteMode.DISK_ONLY)
        print(f'### Wrote key {key}')

      im = (im * 255.0).astype(np.uint8)
      frame = np.zeros((height, width), dtype=np.uint8)
      frame[0:im.shape[0], 0:im.shape[1]] = im

      out.write(frame)

    out.release()
    thumbnail_video = open(video_filename, 'rb').read()

    key = ('stack_subframe_thumbnail_video', stack.stable_hash)
    context.set(key, thumbnail_video, write_mode=WriteMode.DISK_ONLY)
    print(f'### Wrote key {key}')

  return best_light_frame.filename


@ray.remote
def gather() -> None:
  pass
