import ray
import numpy as np

from context import context, WriteMode
from stacks.types import StackHash
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import SimpleStacksDict
from hashing import hash_list
# TODO: Unify quarantining
import quarantining
from previews.operations import resize_array, array_to_png
from file_io.generic import read_image
from previews.types import PreviewHash
from previews.calibration import calibrate_image
from previews.operations import autostretch
from profiling import track_performance
from metadata.classes import ImageMetadata
from dependency_graph.astroflow_node import AstroFlowNode
from dependency_graph import Graph
from stacks.customization import CustomizedStack
from quarantining.settings import StackSettings


# TODO: Pass stack directly and cache this.
@AstroFlowNode()
def create_flat_correction_thumbnails(stack: CustomizedStack) -> Graph:
  calibration_stacks = context.calibration_stacks
  metadata = context.metadata

  # There should be many flat correction thumbnails for every stack, one for
  # each applicable run of flat frames. This key should have a dict for a
  # value. The keys of the dict are the content hashes of the runs of flat
  # frames.
  key = ('flat_correction_thumbnails', stack.stable_hash)
  thumbnails = context.get(key, {})

  flat_stack_ids = [
      id for id, s in calibration_stacks.items() if s.image_type == 'flat' and
      s.camera == stack.camera and s.filter == stack.filter
  ]

  for substack in stack.contents.values():
    # TODO Maybe unify this operation, since it's done everywhere.
    light_frames_by_stars = quarantining.get_light_frames_sorted_by_stars(
        substack.frames)
    light_frame = light_frames_by_stars[0]

    width = context.subframe_thumbnail_width
    assert light_frame.plate_solution is not None
    rotate = light_frame.plate_solution.rotation_angle < 0.0
    light_image = (resize_array(read_image(light_frame.filename),
                                width)).astype(np.float32)

    def first_frame(stack_id: StackHash) -> ImageMetadata:
      return metadata[calibration_stacks[stack_id].frames[0]]

    # TODO: These will probably fail for any light stack that doesn't have
    # calibration images. Add a test case.
    #
    # TODO: Unify this get first subframe and calibration stuff with
    # create_stack_subframe_thumbnails().
    bias_frame = bias_image = None
    if 'bias' in substack.dependencies:
      bias_frame = first_frame(substack.dependencies['bias'])
      bias_image = (resize_array(read_image(bias_frame.filename),
                                 width)).astype(np.float32)

    dark_frame = dark_image = None
    if 'dark' in substack.dependencies:
      dark_frame = first_frame(substack.dependencies['dark'])
      dark_image = (resize_array(read_image(dark_frame.filename),
                                 width)).astype(np.float32)

    for flat_stack_id in flat_stack_ids:
      flat_frame = first_frame(flat_stack_id)
      flat_image = (resize_array(read_image(flat_frame.filename),
                                 width)).astype(np.float32)

      thumbnail = array_to_png(
          autostretch(
              calibrate_image(light_frame, light_image, bias_frame, bias_image,
                              dark_frame, dark_image, flat_frame, flat_image,
                              rotate)))
      hash = PreviewHash(hash_list([substack.content_hash, flat_stack_id]))
      thumbnails[flat_stack_id] = hash

      thumbnail_key = ('flat_correction_thumbnail', hash)
      context.set(thumbnail_key, thumbnail, write_mode=WriteMode.DISK_ONLY)
      print(f'### Wrote key {thumbnail_key}')

    key = ('flat_correction_thumbnails', stack.stable_hash)
    context.set(key, thumbnails, write_mode=WriteMode.DISK_ONLY)
    print(
        f'### Wrote key {key} with {len(thumbnails)} flat-corrected thumbnails')
