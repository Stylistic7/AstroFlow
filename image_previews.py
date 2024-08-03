import numpy as np
from numpy import abs, minimum, maximum, median
import cv2
from skimage.transform import resize
import os
import tempfile
import ray

from context import context
from hashing import hash_string
from file_io.generic import read_image
import subframe_selection
from stacks import stacking
from dependency_graph import Graph
from metadata.actor import get_metadata, get_metadata_for_stack

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

MIN_STAR_FRACTION = 0.5


def resize_array(array, new_width):
  assert not np.isnan(np.min(array))
  height, width = array.shape
  new_height = (height / width) * new_width
  result = resize(array, (int(new_height), int(new_width)))
  assert not np.isnan(
      np.min(result)
  ), f"Failed to resize image dtype {array.dtype} min {np.min(array)} max {np.max(array)}"
  return result


# See this YouTube video: https://www.youtube.com/watch?v=em9IQzbkBuU
# which links to this PixelMath stretching:
# https://drive.google.com/drive/folders/1ZuE8VKlrryofFAGdx-hIH8drrdNZxVCN
def autostretch(image, debug=False):
  image = image.astype(np.float32) / np.max(image)
  unclipped = np.where((image > 0.0) & (image < 1.0))
  if debug:
    print(f'### autostretch, input image '
          f'min {np.min(image)} max {np.max(image)} '
          f'median {np.median(image[unclipped])}')

  curve = 1.0  # 1 = normal stretch, > 1 = arcsin style
  stretch = 0.20  # target background value
  clip = 2.8  # increase to prevent shadow clipping

  # The median of the absolute values of the residuals from the data's median.
  def mdev(image):
    return median(abs(image - median(image)))

  E1 = np.clip(
      median(image[unclipped]) + -clip * 1.4826 * mdev(image[unclipped]), 0, 1)
  E2 = maximum(0, (image - E1) / (1 - E1))
  E3 = median(image[unclipped]) - E1
  E4 = minimum(1, (1 / curve))
  E5 = np.nan_to_num(
      ((stretch - 1) * E3) / ((2 * stretch - 1) * pow(E3, E4) - stretch))
  output = np.nan_to_num(((E5 - 1) * E2) / ((2 * E5 - 1) * pow(E2, E4) - E5))

  if debug:
    print(f'### E1 {E1}')
    print(f'### E2 {E2}')
    print(f'### E3 {E3}')
    print(f'### E4 {E4}')
    print(f'### E5 {E5}')

  assert not np.isnan(output).any(), 'stretched image contains NaN'

  if debug:
    print(f'### autostretch, output image '
          f'min {np.min(output)} max {np.max(output)}')
  return output


def array_to_png(array, debug=False):
  if np.issubdtype(array.dtype, np.floating):
    assert not np.isnan(array).any(), 'array contains NaN'

  array = (array * 255.0).astype(np.uint8)
  if debug:
    print(f'### array_to_png, array min {np.min(array)} max {np.max(array)}')
  _, png = cv2.imencode('.png', array)
  return png.tobytes()


# TODO: Average flat frames before applying them.
def _make_thumbnail(light_frame, light_image, bias_frame, bias_image,
                    dark_frame, dark_image, flat_frame, flat_image, rotate):

  if bias_image is None or dark_image is None or flat_image is None:
    print('### no calibration images')
    if rotate:
      light_image = np.rot90(light_image, 2)
    return autostretch(light_image)

  image_sizes = {
      light_image.shape, dark_image.shape, bias_image.shape, flat_image.shape
  }
  assert len(image_sizes) == 1, (
      f"Shapes don't match: light {light_image.shape}, "
      f"bias {bias_image.shape}, "
      f"dark {dark_image.shape}, "
      f"flat {flat_image.shape}")

  light_dark_scale = light_frame.header['EXPOSURE'] / dark_frame.header[
      'EXPOSURE']
  flat_dark_scale = flat_frame.header['EXPOSURE'] / dark_frame.header['EXPOSURE']

  image = (light_image -
           bias_image) - light_dark_scale * (dark_image - bias_image)
  flat_image = (flat_image -
                bias_image) - flat_dark_scale * (dark_image - bias_image)
  image = np.divide(image,
                    flat_image,
                    out=np.zeros_like(image),
                    where=flat_image != 0)

  if rotate:
    image = np.rot90(image, 2)

  return autostretch(np.clip(image, 0.0, 1.0))


@ray.remote(resources={'fastest_storage': 0.001})
def scatter_stack_thumbnails(mode):
  stacks = context.stacks
  cache_keys = context.get_cache_keys()

  if mode == 'subframe':
    subframe_images = [(create_flat_correction_thumbnails, id)
                       for id, s in stacks.items()
                       if s.image_type == 'light' and not s.stable_hash]
    subframe_videos = [
        (create_stack_subframe_thumbnails, id)
        for id, s in stacks.items()
        if s.image_type == 'light' and s.stable_hash and (
            ('stack_subframe_thumbnail_image', id) not in cache_keys or
            ('stack_subframe_thumbnail_video', id) not in cache_keys)
    ]
    tasks = subframe_images + subframe_videos
  elif mode == 'master':
    tasks = [(create_stack_master_thumbnail, id)
             for id, s in stacks.items()
             if s.image_type == 'light' and s.stable_hash and
             ('stack_master_thumbnail_image', id) not in cache_keys]

  return Graph({gather: tasks})


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

  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.frames)
  light_frame = light_frames_by_stars[0]

  width = context.subframe_thumbnail_width
  rotate = light_frame.plate_solution.rotation_angle < 0.0
  light_image = (resize_array(read_image(light_frame.filename),
                              width)).astype(np.float32)

  metadata = get_metadata_for_stack(stack_id)

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
        _make_thumbnail(light_frame, light_image, bias_frame, bias_image,
                        dark_frame, dark_image, flat_frame, flat_image, rotate))
    hash = hash_string(thumbnail)
    thumbnails[flat_stack_id] = hash

    key = ('flat_correction_thumbnail', hash)
    context.set(key, thumbnail, flush=True)
    print(f'### Wrote key {key}')

  key = ('flat_correction_thumbnails', stack_id)
  context.set(key, thumbnails, flush=True)
  print(f'### Wrote key {key} with {len(thumbnails)} flat-corrected thumbnails')


@ray.remote
def create_stack_subframe_thumbnails(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]
  print(f'Entering create_stack_subframe_thumbnails() for stack {stack}')

  # Don't make previews for calibration stacks.
  assert stack.stable_hash

  settings_key = ('stack_settings', stack_id)
  settings = context[settings_key] or {}
  # FIXME the dependencies have to be taken from the substacks, e.g. the
  # light runs. But the sorting by number of stars makes it hard to look up
  # the substack it belonged to.
  # dependencies = stack.dependencies | settings

  width = context.subframe_thumbnail_width

  metadata = get_metadata_for_stack(stack_id)

  # Cache the calibration images so they're only read once.
  first_images = {}

  def get_first_image(stack_id):
    if not stack_id:
      return None
    elif stack_id in first_images:
      return first_images[stack_id]
    else:
      frame = metadata[stacks[stack_id].frames[0]]
      image = resize_array(read_image(frame.filename), width).astype(np.float32)
      first_images[stack_id] = (frame, image)
      return first_images[stack_id]

  # Now sort all the light frames (no matter which run they're from) by the
  # number of stars. The better images should be in front.
  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.all_frames())

  best_light_frame = light_frames_by_stars[0]
  image = resize_array(read_image(best_light_frame.filename), width)
  height, width = image.shape

  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(light_frames_by_stars)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)
    for i, light_frame in enumerate(light_frames_by_stars):
      rotate = light_frame.plate_solution.rotation_angle < 0.0

      light_image = resize_array(read_image(light_frame.filename),
                                 width).astype(np.float32)

      # TODO: This feels a little hacky.
      # FIXME and it isn't working
      dependencies = settings
      if substack_id := next(
          iter(s for s in stack.contents
               if light_frame.content_hash in stacks[s].frames), None):
        dependencies = stacks[substack_id].dependencies | settings

      bias_frame = bias_image = None
      if 'bias' in dependencies:
        bias_frame, bias_image = get_first_image(dependencies['bias'])

      dark_frame = dark_image = None
      if 'dark' in dependencies:
        dark_frame, dark_image = get_first_image(dependencies['dark'])

      flat_frame = flat_image = None
      if 'flat' in dependencies:
        flat_frame, flat_image = get_first_image(dependencies['flat'])

      im = _make_thumbnail(light_frame, light_image, bias_frame, bias_image,
                           dark_frame, dark_image, flat_frame, flat_image,
                           rotate)

      if i == 0:
        key = ('stack_subframe_thumbnail_image', stack_id)
        context.set(key, array_to_png(im), flush=True)
        print(f'### Wrote key {key}')

      im = (im * 255.0).astype(np.uint8)
      frame = np.zeros((height, width), dtype=np.uint8)
      frame[0:im.shape[0], 0:im.shape[1]] = im

      out.write(frame)

    out.release()
    thumbnail_video = open(video_filename, 'rb').read()

    key = ('stack_subframe_thumbnail_video', stack_id)
    context.set(key, thumbnail_video, flush=False)
    print(f'### Wrote key {key}')


@ray.remote
def gather():
  pass
