import ray
import numpy as np

from context import context
from hashing import hash_string
from metadata.actor import get_metadata_for_stack
import subframe_selection
from previews.operations import resize_array, array_to_png
from file_io.generic import read_image
from previews.thumbnail import make_thumbnail


# TODO: Is it only necessary to make the thumbnails for the flat stacks
# immediately before and immediately after the light frames? I suppose people
# could make bad flat frames, or confuse them somehow, and want all of the
# flexibility.
@ray.remote
def create_flat_correction_thumbnails(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

  # There should be many flat correction thumbnails for every stack, one for
  # each applicable run of flat frames. This key should have a dict for a
  # value. The keys of the dict are the content hashes of the runs of flat
  # frames.
  key = ('flat_correction_thumbnails', stack_id)
  thumbnails = context.get(key, {})

  flat_stack_ids = [
      id for id, s in stacks.items()
      if not s.stable_hash and s.image_type == 'flat' and
      s.camera == stack.camera and s.filter == stack.filter
  ]

  if all(f in thumbnails for f in flat_stack_ids):
    return

  assert not stack.stable_hash

  metadata = get_metadata_for_stack(stack_id)

  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.frames, metadata)
  light_frame = light_frames_by_stars[0]

  width = context.subframe_thumbnail_width
  rotate = light_frame.plate_solution.rotation_angle < 0.0
  light_image = (resize_array(read_image(light_frame.filename),
                              width)).astype(np.float32)

  def first_frame(stack_id):
    return metadata[stacks[stack_id].frames[0]]

  settings_key = ('stack_settings', stack_id)
  settings = context[settings_key] or {}
  dependencies = stack.dependencies | settings

  # TODO: These will probably fail for any light stack that doesn't have
  # calibration images. Add a test case.
  bias_frame = bias_image = None
  if 'bias' in dependencies:
    bias_frame = first_frame(dependencies['bias'])
    bias_image = (resize_array(read_image(bias_frame.filename),
                               width)).astype(np.float32)

  dark_frame = dark_image = None
  if 'dark' in dependencies:
    dark_frame = first_frame(dependencies['dark'])
    dark_image = (resize_array(read_image(dark_frame.filename),
                               width)).astype(np.float32)

  for flat_stack_id in flat_stack_ids:
    if flat_stack_id in thumbnails:
      continue

    flat_frame = first_frame(flat_stack_id)
    flat_image = (resize_array(read_image(flat_frame.filename),
                               width)).astype(np.float32)

    thumbnail = array_to_png(
        make_thumbnail(light_frame, light_image, bias_frame, bias_image,
                       dark_frame, dark_image, flat_frame, flat_image, rotate))
    hash = hash_string(thumbnail)
    thumbnails[flat_stack_id] = hash

    key = ('flat_correction_thumbnail', hash)
    context.set(key, thumbnail, flush=True)
    print(f'### Wrote key {key}')

  key = ('flat_correction_thumbnails', stack_id)
  context.set(key, thumbnails, flush=True)
  print(f'### Wrote key {key} with {len(thumbnails)} flat-corrected thumbnails')
