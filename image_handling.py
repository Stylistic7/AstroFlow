import numpy as np
from numpy import abs, minimum, maximum, median
import cv2
import os
import tempfile
from multiprocessing import Pool
import ray

import ray_diskcache
from cache import cache
import file_io
import runs
import hashing

MIN_STAR_FRACTION = 0.5


def read_image(filename):
  if filename.endswith('.xisf'):
    return file_io.read_xisf_image(filename)
  elif filename.endswith('.fit') or filename.endswith('.fits'):
    return file_io.read_fits_image(filename)


def resize_array(array, new_width):
  height, width = array.shape
  new_height = (height / width) * new_width
  return cv2.resize(array, (int(new_width), int(new_height)),
                    interpolation=cv2.INTER_AREA)


def autostretch(image):
  # FIXME what max to use here?
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


def make_thumbnail(image_hashes,
                   fits_headers,
                   light_frame,
                   dark_frame,
                   bias_frame,
                   flat_frame,
                   rotate,
                   width=1200):
  light_image = resize_array(read_image(image_hashes[light_frame]), width)

  if rotate:
    light_image = np.rot90(light_image, 2)

  if not dark_frame or not bias_frame or not flat_frame:
    return autostretch(light_image)

  dark_image = resize_array(read_image(image_hashes[dark_frame]), width)
  bias_image = resize_array(read_image(image_hashes[bias_frame]), width)
  flat_image = resize_array(read_image(image_hashes[flat_frame]), width)

  image_sizes = {
      light_image.shape, dark_image.shape, bias_image.shape, flat_image.shape
  }
  assert (len(image_sizes) == 1), (
      f"Shapes don't match: {image_hashes[light_frame]} {light_image.shape}, "
      f"{image_hashes[dark_frame]} {dark_image.shape}, "
      f"{image_hashes[bias_frame]} {bias_image.shape}, "
      f"{image_hashes[flat_frame]} {flat_image.shape}")

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


# def _get_flat_frame_for_light_frame(light_frame, stack, fits_headers):
#   for light_run, flat_run in stack.flat_frame_associations:
#     light_frames = runs.get_frames_for_run(light_run, stack.light_frames,
#                                            fits_headers)
#     if light_frame not in light_frames:
#       continue

#     flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
#                                           fits_headers)
#     if flat_frames:
#       return flat_frames[0]


@ray.remote
def scatter_stack_thumbnails(context):
  stacks = ray.get(context.stacks)

  def has_thumbnails(stack):
    keys = [
        ('stack_thumbnail_image', stack.hash()),
        ('stack_thumbnail_video', stack.hash()),
    ]
    return all(ray_diskcache.contains(k) for k in keys)

  ray.get([
      compute_stack_thumbnails.remote(context, stack)
      for stack in stacks.values()
      if not has_thumbnails(stack)
  ])

  return context


# TODO: Put the thumbnail size in a user config.
@ray.remote(scheduling_strategy="SPREAD",
            num_cpus=0.75,
            max_retries=5,
            retry_exceptions=True)
def compute_stack_thumbnails(context, stack, width=1200):
  print(f'Entering compute_stack_thumbnails() for stack {str(stack)}')
  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)

  # TODO: User masters when available.
  light_frames = [
      f for run in stack.runs for f in run.frames
      if f in fits_headers and fits_headers[f]['IMAGETYP'] == 'Light'
  ]
  dark_frames = [
      f for run in stack.runs for f in run.frames
      if f in fits_headers and fits_headers[f]['IMAGETYP'] == 'Dark'
  ]
  bias_frames = [
      f for run in stack.runs for f in run.frames
      if f in fits_headers and fits_headers[f]['IMAGETYP'] == 'Bias'
  ]
  flat_frames = [
      f for run in stack.runs for f in run.frames
      if f in fits_headers and fits_headers[f]['IMAGETYP'] == 'Flat'
  ]

  # TODO: Split the try into another function.
  try:
    light_frames_by_stars = sorted(light_frames,
                                   key=lambda f: plate_solutions[f].num_stars,
                                   reverse=True)

    image = resize_array(read_image(image_hashes[light_frames_by_stars[0]]),
                         width)
    height, width = image.shape

    height = int(((height + 8 - 1) / 8) * 8)
    print(f'Creating video of width {width}, height {height} '
          f'with {len(light_frames)} frames')

    with tempfile.TemporaryDirectory() as tmp_dir:
      fourcc = cv2.VideoWriter_fourcc(*'VP80')
      video_filename = os.path.join(tmp_dir, 'output.webm')
      out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height),
                            False)
      made_stack_thumbnail = False
      for light_frame in light_frames_by_stars:
        # flat_frame = _get_flat_frame_for_light_frame(light_frame, stack,
        #                                              fits_headers)
        flat_frame = flat_frames[0] if len(flat_frames) else None
        dark_frame = dark_frames[0] if len(dark_frames) else None
        bias_frame = bias_frames[0] if len(bias_frames) else None
        rotate = plate_solutions[light_frame].rotation_angle < 0.0
        try:
          im = make_thumbnail(image_hashes, fits_headers, light_frame,
                              dark_frame, bias_frame, flat_frame, rotate)
        except Exception as e:
          print(f'make_thumbnail() failed for {image_hashes[light_frame]} '
                f'due to exception {e}')
          continue

        if not made_stack_thumbnail:
          key = ('stack_thumbnail_image', stack.hash())
          ray_diskcache.set(key, array_to_png(im), keep_in_memory=False)
          print(f'Wrote key {key}')
          made_stack_thumbnail = True

        im = (im * 255.0).astype(np.uint8)
        frame = np.zeros((height, width), dtype=np.uint8)
        frame[0:im.shape[0], 0:im.shape[1]] = im

        out.write(frame)

      out.release()
      thumbnail_video = open(video_filename, 'rb').read()

      key = ('stack_thumbnail_video', stack.hash())
      ray_diskcache.set(key, thumbnail_video, keep_in_memory=False)
      print(f'Wrote key {key}')
  except Exception as e:
    print(f'compute_stack_thumbnails failed with exception {e}')


# # TODO: Compare various ways of getting "the best" thumbnail (preview)
# # image for a stack: highest std, greatest star count, etc.
# #
# # Make a library for picking / sorting images by quality.
# def make_and_cache_stack_thumbnails(stack,
#                                     image_hashes,
#                                     plate_solutions,
#                                     width=1200):
#   height, width = resize_array(
#       file_io.read_fits_image(image_hashes[stack.light_frames[0]]),
#       width).shape

#   height = int(((height + 8 - 1) / 8) * 8)
#   print(f'Creating video of width {width}, height {height} '
#         f'with {len(stack.light_frames)} frames')

#   with tempfile.TemporaryDirectory() as tmp_dir:
#     fourcc = cv2.VideoWriter_fourcc(*'VP80')
#     video_filename = os.path.join(tmp_dir, 'output.webm')
#     out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)

#     light_frames_by_stars = sorted(stack.light_frames,
#                                    key=lambda f: plate_solutions[f].num_stars,
#                                    reverse=True)
#     thumbnail_image = array_to_png(
#         cache[('thumbnail_image', light_frames_by_stars[0])])

#     for i, hash in enumerate(light_frames_by_stars):
#       im = cache[('thumbnail_image', hash)]
#       im = (im * 255.0).astype(np.uint8)

#       frame = np.zeros((height, width), dtype=np.uint8)
#       frame[0:im.shape[0], 0:im.shape[1]] = im

#       out.write(frame)

#     out.release()
#     thumbnail_video = open(video_filename, 'rb').read()

#     cache[('stack_thumbnail_image', stack.id)] = thumbnail_image
#     cache[('stack_thumbnail_video', stack.id)] = thumbnail_video
