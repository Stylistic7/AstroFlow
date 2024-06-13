import numpy as np
from numpy import abs, minimum, maximum, median
import cv2
import os
import tempfile
import ray

from context import context
import file_io
from hashing import hash_string
from file_io import read_image
import subframe_selection
import stacking
from dependency_graph import Graph

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
  height, width = array.shape
  new_height = (height / width) * new_width
  return cv2.resize(array, (int(new_width), int(new_height)),
                    interpolation=cv2.INTER_AREA)


def autostretch(image):
  # TODO: what max to use here?
  image = image.astype(np.float32) / 65536.0

  curve = 1.0
  stretch = 0.20

  def mdev(image):
    return median(abs(image - median(image)))

  E1 = minimum(maximum(0.0, median(image) + -2.8 * 1.4826 * mdev(image)), 1.0)
  E2 = maximum(0.0, (image - E1) / (1 - E1))
  E3 = median(image) - E1
  E4 = minimum(1.0, (1.0 / curve))
  E5 = ((stretch - 1.0) * E3) / ((2.0 * stretch - 1.0) * pow(E3, E4) - stretch)
  E6 = ((E5 - 1.0) * E2) / ((2.0 * E5 - 1.0) * pow(E2, E4) - E5)

  return E6


def array_to_png(array):
  array = (array * 255.0).astype(np.uint8)
  _, png = cv2.imencode('.png', array)
  return png.tobytes()


# TODO: Average flat frames before applying them.
def _make_thumbnail(fits_headers, light_frame, light_image, bias_frame,
                    bias_image, dark_frame, dark_image, flat_frame, flat_image,
                    rotate):

  if bias_image is None or dark_image is None or flat_image is None:
    return autostretch(light_image)

  image_sizes = {
      light_image.shape, dark_image.shape, bias_image.shape, flat_image.shape
  }
  assert len(image_sizes) == 1, (
      f"Shapes don't match: light {light_image.shape}, "
      f"bias {bias_image.shape}, "
      f"dark {dark_image.shape}, "
      f"flat {flat_image.shape}")

  light_dark_scale = fits_headers[light_frame]['EXPOSURE'] / fits_headers[
      dark_frame]['EXPOSURE']
  flat_dark_scale = fits_headers[flat_frame]['EXPOSURE'] / fits_headers[
      dark_frame]['EXPOSURE']

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


@ray.remote
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
        if s.image_type == 'light' and s.stable_hash and
        ('stack_subframe_thumbnail_image', id) not in cache_keys and
        ('stack_subframe_thumbnail_video', id) not in cache_keys
    ]
    tasks = subframe_images + subframe_videos
  elif mode == 'master':
    tasks = [(create_stack_master_thumbnail, id)
             for id, s in stacks.items()
             if s.image_type == 'light' and s.stable_hash and
             ('stack_master_thumbnail_image', id) not in cache_keys]

  return Graph({gather: tasks})


# TODO: This function needs to decend into the Stack contents and get
# calibrations at the lower level.
def _get_calibrations(stack, stacks):
  assert stack.image_type == 'light'

  calibrations = {}
  bias_frame = dark_frame = flat_frame = None
  if bias_stack := stack.dependencies.get('bias', None):
    bias_frame = stacks[bias_stack].frames[0]
  if dark_stack := stack.dependencies.get('dark', None):
    dark_frame = stacks[dark_stack].frames[0]
  if flat_stack := stack.dependencies.get('flat', None):
    flat_frame = stacks[flat_stack].frames[0]

  for light_frame in stack.frames:
    calibrations[light_frame] = dict(bias_frame=bias_frame,
                                     dark_frame=dark_frame,
                                     flat_frame=flat_frame)

  for c in stack.contents:
    calibrations |= _get_calibrations(stacks[c], stacks)

  return calibrations


# TODO: This runs on every missing master thumbnail, even if the masters don't
# exist yet. It takes up time and is unnecessary. Find some way to cache this
# result or speed it up.
@ray.remote
def create_stack_master_thumbnail(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]
  print(f'Entering create_stack_master_thumbnail() for stack {stack}')

  image_hashes = context.image_hashes
  fits_headers = context.fits_headers

  # Find the master(s).
  masters = stacking.find_matching_masters(stack, image_hashes, fits_headers)
  print(f'Found masters {masters}')

  if not masters:
    return

  if len(masters) > 1:
    print(f'Found more than one master with hash {stack_id}')

  master = masters[0]

  # Read the master image and autostretch it.
  image = autostretch(
      resize_array(read_image(image_hashes[master]),
                   context.master_thumbnail_width))

  # Store the thumbnail.
  key = ('stack_master_thumbnail_image', stack_id)
  context.set(key, array_to_png(image), flush=True)
  print(f'Wrote key {key}')

  return key


# TODO: Is it only necessary to make the thumbnails for the flat stacks
# immediately before and immediately after the light frames? I suppose people
# could make bad flat frames, or confuse them somehow, and want all of the
# flexibility.
@ray.remote
def create_flat_correction_thumbnails(stack_id):
  stacks = context.stacks
  stack = stacks[stack_id]

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
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers
  plate_solutions = context.plate_solutions

  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.frames, plate_solutions)
  light_frame, _ = light_frames_by_stars[0]

  width = context.subframe_thumbnail_width
  rotate = plate_solutions[light_frame].rotation_angle < 0.0
  light_image = resize_array(read_image(image_hashes[light_frame]), width)

  def first_frame(stack_id):
    return stacks[stack_id].frames[0]

  settings_key = ('stack_settings', stack_id)
  settings = context.settings_key or {}
  dependencies = stack.dependencies | settings

  # TODO: These will probably fail for any light stack that doesn't have
  # calibration images. Add a test case.
  bias_frame = bias_image = None
  if 'bias' in dependencies:
    bias_frame = first_frame(dependencies['bias'])
    bias_image = resize_array(read_image(image_hashes[bias_frame]), width)

  dark_frame = dark_image = None
  if 'dark' in dependencies:
    dark_frame = first_frame(dependencies['dark'])
    dark_image = resize_array(read_image(image_hashes[dark_frame]), width)

  for flat_stack_id in flat_stack_ids:
    if flat_stack_id in thumbnails:
      continue

    flat_frame = first_frame(flat_stack_id)
    flat_image = resize_array(read_image(image_hashes[flat_frame]), width)

    thumbnail = array_to_png(
        _make_thumbnail(fits_headers, light_frame, light_image, bias_frame,
                        bias_image, dark_frame, dark_image, flat_frame,
                        flat_image, rotate))
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

  image_hashes = context.image_hashes
  fits_headers = context.fits_headers
  plate_solutions = context.plate_solutions
  width = context.subframe_thumbnail_width

  # Don't make previews for calibration stacks.
  assert stack.stable_hash

  calibrations = _get_calibrations(stack, stacks)

  # Cache the calibration images so they're only read once.
  calibration_images = {}

  def get_calibration_image(hash):
    if not hash:
      return None
    elif hash in calibration_images:
      return calibration_images[hash]
    else:
      image = resize_array(read_image(image_hashes[hash]), width)
      calibration_images[hash] = image
      return image

  # Now sort all the light frames (no matter which run they're from) by the
  # number of stars. The better images should be in front.
  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack.all_frames(stacks), plate_solutions)

  best_light_frame, _ = light_frames_by_stars[0]
  image = resize_array(read_image(image_hashes[best_light_frame]), width)
  height, width = image.shape

  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(light_frames_by_stars)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)
    for i, (light_frame, _) in enumerate(light_frames_by_stars):
      rotate = plate_solutions[light_frame].rotation_angle < 0.0

      light_image = resize_array(read_image(image_hashes[light_frame]), width)

      bias_frame = calibrations[light_frame]['bias_frame']
      bias_image = get_calibration_image(bias_frame)

      dark_frame = calibrations[light_frame]['dark_frame']
      dark_image = get_calibration_image(dark_frame)

      flat_frame = calibrations[light_frame]['flat_frame']
      flat_image = get_calibration_image(flat_frame)

      im = _make_thumbnail(fits_headers, light_frame, light_image, bias_frame,
                           bias_image, dark_frame, dark_image, flat_frame,
                           flat_image, rotate)

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
def gather(context):
  pass
