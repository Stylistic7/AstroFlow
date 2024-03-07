import numpy as np
from numpy import abs, minimum, maximum, median
import cv2
import os
import tempfile
import ray

import ray_diskcache
import file_io
from file_io import read_image
import subframe_selection
import stacking

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


def _make_thumbnail(fits_headers, light_frame, light_image, bias_frame,
                    bias_image, dark_frame, dark_image, flat_frame, flat_image,
                    rotate):

  if rotate:
    light_image = np.rot90(light_image, 2)

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

  return autostretch(np.clip(image, 0.0, 1.0))


@ray.remote
def scatter_stack_thumbnails(context, mode):
  fits_headers = ray.get(context.fits_headers)
  stacks = ray.get(context.stacks)

  def has_thumbnail(stack):
    geometric_hash = stack.geometric_hash()
    key = ('stack_thumbnail_image', mode, geometric_hash)
    return ray_diskcache.contains(key)

  to_compute = [
      s for s in stacks.values() if s.geometric_hash() and not has_thumbnail(s)
  ]
  print(f'### Need to compute thumbnails for {len(to_compute)} stacks')

  waitables = [
      compute_stack_thumbnails.remote(context, stack, mode)
      for stack in to_compute
  ]

  while len(waitables):
    result, waitables = ray.wait(waitables)
    try:
      ray.get(result)
    except Exception as e:
      print(f'### compute_stack_thumbnails threw exception {e}')

  return context


def _get_calibrations(stack, stacks, fits_headers):
  calibrations = {}
  for run in stack.runs:
    # Find the first bias, dark, and flat frame in the run's dependencies.
    dependencies = [stacks[d].runs[0].frames[0] for d in run.stack_dependencies]
    bias_frame = next(
        (d for d in dependencies if file_io.is_bias_frame(fits_headers[d])),
        None)
    dark_frame = next(
        (d for d in dependencies if file_io.is_dark_frame(fits_headers[d])),
        None)
    flat_frame = next(
        (d for d in dependencies if file_io.is_flat_frame(fits_headers[d])),
        None)

    for light_frame in run.frames:
      calibrations[light_frame] = dict(bias_frame=bias_frame,
                                       dark_frame=dark_frame,
                                       flat_frame=flat_frame)

  return calibrations


# TODO: Put the thumbnail sizes in a user config.
@ray.remote(scheduling_strategy="SPREAD", max_retries=5, retry_exceptions=True)
def compute_stack_thumbnails(context, stack, mode):
  if mode == 'subframe':
    return compute_stack_subframe_thumbnails(context, stack, width=1200)
  elif mode == 'master':
    return compute_stack_master_thumbnail(context, stack, width=2160)
  elif mode == 'post_processed':
    print('### not implemented')


def compute_stack_master_thumbnail(context, stack, width=2160):
  print(f'Entering compute_stack_master_thumbnail() for stack {str(stack)}')
  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)

  content_hash = stack.content_hash()

  # Find the master(s).
  masters = stacking.find_matching_masters(image_hashes, fits_headers,
                                           content_hash)
  print(f'### Found masters {masters}')

  if not masters:
    return

  if len(masters) > 1:
    print(f'### Found more than one master with hash {content_hash}')

  master = masters[0]

  # Read the master image and autostretch it.
  image = autostretch(resize_array(read_image(image_hashes[master]), width))

  # Store the thumbnail.
  key = ('stack_thumbnail_image', 'master', content_hash)
  ray_diskcache.set(key, array_to_png(image), keep_in_memory=False)
  print(f'Wrote key {key}')


def compute_stack_subframe_thumbnails(context, stack, width):
  print(f'Entering compute_stack_subframe_thumbnails() for stack {str(stack)}')
  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)

  geometric_hash = stack.geometric_hash()

  # Don't make previews for calibration stacks.
  light_frames = [f for run in stack.runs for f in run.frames]
  if not all(file_io.is_light_frame(fits_headers[f]) for f in light_frames):
    print(f'Ignoring {stack} because it is not composed entirely '
          f'of light frames.')
    return

  calibrations = _get_calibrations(stack, stacks, fits_headers)

  # Now sort all the light frames (no matter which run they're from) by the
  # number of stars. The better images should be in front.
  light_frames_by_stars = subframe_selection.get_light_frames_sorted_by_stars(
      stack, plate_solutions)

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

  best_light_frame, _ = light_frames_by_stars[0]
  image = resize_array(read_image(image_hashes[best_light_frame]), width)
  height, width = image.shape

  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(light_frames)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)
    made_stack_thumbnail = False
    for light_frame, _ in light_frames_by_stars:
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

      if not made_stack_thumbnail:
        key = ('stack_thumbnail_image', 'subframe', geometric_hash)
        ray_diskcache.set(key, array_to_png(im), keep_in_memory=False)
        print(f'Wrote key {key}')
        made_stack_thumbnail = True

      im = (im * 255.0).astype(np.uint8)
      frame = np.zeros((height, width), dtype=np.uint8)
      frame[0:im.shape[0], 0:im.shape[1]] = im

      out.write(frame)

    out.release()
    thumbnail_video = open(video_filename, 'rb').read()

    key = ('stack_thumbnail_video', geometric_hash)
    ray_diskcache.set(key, thumbnail_video, keep_in_memory=False)
    print(f'Wrote key {key}')
