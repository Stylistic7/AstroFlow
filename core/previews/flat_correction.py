from context import context, WriteMode
from previews.operations import array_to_png
from previews.calibration import calibrate_subframe
from previews.operations import autostretch
from decorators.task_decorator import task
from dependency_graph import Graph
from customization.types import CustomizedStack
from customization.subframe_selection import get_light_frames_sorted_by_stars


@task.remote
def create_flat_correction_thumbnails(stack: CustomizedStack) -> None:
  calibration_stacks = context.calibration_stacks

  flat_stack_ids = [
      id for id, s in calibration_stacks.items() if s.image_type == 'flat' and
      s.camera == stack.camera and s.filter == stack.filter
  ]

  light_frames_by_stars = get_light_frames_sorted_by_stars(stack)

  for substack in stack.contents.values():
    light_frame = stack.all_metadata()[next(
        l for l in light_frames_by_stars if l in substack.frames)]

    width = context.subframe_thumbnail_width

    for flat_stack_id in flat_stack_ids:
      dependencies = substack.dependencies.copy()
      dependencies['flat'] = flat_stack_id

      thumbnail = array_to_png(
          autostretch(calibrate_subframe(light_frame, dependencies, width)))

      thumbnail_key = ('flat_correction_thumbnail', substack.content_hash,
                       flat_stack_id)
      context.set(thumbnail_key, thumbnail, write_mode=WriteMode.DISK_ONLY)
      print(f'### Wrote key {thumbnail_key}')
