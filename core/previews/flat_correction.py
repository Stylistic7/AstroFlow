import tempfile
from pathlib import Path

from context import context
from previews.calibration import calibrate_subframe
from previews.operations import autostretch
from decorators import cached, measure_performance
from customization.types import CustomizedStack
from customization.subframe_selection import get_light_frames_sorted_by_stars
from decorators import create_thumbnail
from stacks.simple_stack import SimpleStack
from file_io.tiff_file_io import write_tiff_image
from products import makes_image


@makes_image(SimpleStack).as_subframe()
@measure_performance
@create_thumbnail
def flat_corrected_stack(
    stack: CustomizedStack, substack: SimpleStack,
    flat_stack: SimpleStack) -> tempfile.NamedTemporaryFile:
  width = context.subframe_thumbnail_width

  light_frames_by_stars = get_light_frames_sorted_by_stars(stack)
  light_frame = stack.all_metadata()[next(
      l for l in light_frames_by_stars if l in substack.frames)]

  dependencies = substack.dependencies.copy()
  dependencies['flat'] = flat_stack.content_hash

  temp_file = tempfile.NamedTemporaryFile(suffix=".tiff", delete=True)
  write_tiff_image(
      Path(temp_file.name),
      autostretch(calibrate_subframe(light_frame, dependencies, width)))

  return temp_file


@cached.remote
def create_flat_correction_thumbnails(stack: CustomizedStack) -> None:
  calibration_stacks = context.calibration_stacks

  flat_stack_ids = [
      id for id, s in calibration_stacks.items() if s.image_type == 'flat' and
      s.camera == stack.camera and s.filter == stack.filter
  ]

  for substack in stack.contents.values():
    for flat_stack_id in flat_stack_ids:
      flat_corrected_stack(stack, substack, calibration_stacks[flat_stack_id])
