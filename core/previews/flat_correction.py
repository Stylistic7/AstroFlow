import ray
import numpy as np

from context import context, WriteMode
from previews.operations import array_to_png
from previews.calibration import calibrate_subframe
from previews.operations import autostretch
from dependency_graph.astroflow_node import AstroFlowNode
from dependency_graph import Graph
from customization.types import CustomizedStack
from customization.subframe_selection import get_light_frames_sorted_by_stars


@AstroFlowNode(cache_result=True)
def create_flat_correction_thumbnails(stack: CustomizedStack) -> Graph:
  calibration_stacks = context.calibration_stacks

  flat_stack_ids = [
      id for id, s in calibration_stacks.items() if s.image_type == 'flat' and
      s.camera == stack.camera and s.filter == stack.filter
  ]

  for substack in stack.contents.values():
    # TODO Maybe unify this operation, since it's done everywhere.
    light_frames_by_stars = get_light_frames_sorted_by_stars(substack.frames)
    light_frame = light_frames_by_stars[0]

    width = context.subframe_thumbnail_width

    for flat_stack_id in flat_stack_ids:
      thumbnail = array_to_png(
          autostretch(
              calibrate_subframe(light_frame, substack.dependencies, width)))
      thumbnail_key = ('flat_correction_thumbnail', substack.content_hash,
                       flat_stack_id)
      context.set(thumbnail_key, thumbnail, write_mode=WriteMode.DISK_ONLY)
      print(f'### Wrote key {thumbnail_key}')
