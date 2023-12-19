from xisf import XISF
import astropy.io.fits as pyfits
import numpy as np
from numpy import abs, minimum, maximum, median
import cv2
import os
import io
from shapely.geometry import Polygon
import tempfile
from multiprocessing import Pool
from functools import partial
import traceback

import file_io

from cache import cache

MIN_STAR_FRACTION = 0.5


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


def _get_flat_frame_for_light_frame(light_frame, stack, fits_headers):
  for light_frames, flat_frames in stack.flat_frame_associations:
    # print(f'### association between {len(light_frames)} light frames '
    #       f'and {len(flat_frames)} flat frames')
    assert len(light_frames)
    assert len(flat_frames)
    if light_frame in light_frames:
      return flat_frames[0]


def _read_image(filename):
  if filename.endswith('.xisf'):
    return file_io.read_xisf_image(filename)
  elif filename.endswith('.fit') or filename.endswith('.fits'):
    return file_io.read_fits_image(filename)


def make_and_cache_thumbnail(light_frame,
                             dark_frame,
                             flat_frame,
                             rotate,
                             fudge=None,
                             width=1200):
  try:
    light_frame_image = resize_array(_read_image(light_frame), width)
    if rotate:
      light_frame_image = np.rot90(light_frame_image, 2)

    if not dark_frame or not flat_frame:
      cache[('thumbnail_image', light_frame)] = autostretch(light_frame_image)
      return

    flat_frame_image = resize_array(_read_image(flat_frame), width)
    dark_frame_image = resize_array(_read_image(dark_frame), width)

    light_frame_image -= dark_frame_image
    flat_frame_image -= dark_frame_image

    # if fudge:
    #   min, max = np.min(flat_frame_image), np.max(flat_frame_image)
    #   flat_frame_image = np.interp(flat_frame_image, [min, max],
    #                                [min + fudge * (max - min), max])

    light_frame_image = np.divide(light_frame_image,
                                  flat_frame_image,
                                  out=np.zeros_like(light_frame_image),
                                  where=flat_frame_image != 0)

    light_frame_image = autostretch(np.clip(light_frame_image, 0.0, 1.0))
    cache[('thumbnail_image', light_frame)] = light_frame_image
  except Exception:
    traceback.print_exc()


# TODO: Compare various ways of getting "the best" thumbnail (preview)
# image for a stack: highest std, greatest star count, etc.
#
# Make a library for picking / sorting images by quality.
def make_and_cache_stack_thumbnails(stack, plate_solutions, width=1200):
  height, width = resize_array(file_io.read_fits_image(stack.light_frames[0]),
                               width).shape

  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(stack.light_frames)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)

    light_frames_by_stars = sorted(stack.light_frames,
                                   key=lambda f: plate_solutions[f].num_stars,
                                   reverse=True)
    thumbnail_image = cache[('thumbnail_image', light_frames_by_stars[0])]

    for i, filename in enumerate(light_frames_by_stars):
      im = cache[('thumbnail_image', filename)]
      im = (im * 255.0).astype(np.uint8)

      frame = np.zeros((height, width), dtype=np.uint8)
      frame[0:im.shape[0], 0:im.shape[1]] = im

      out.write(frame)

    out.release()
    thumbnail_video = open(video_filename, 'rb').read()

    cache[('stack_thumbnail_image', stack.id)] = thumbnail_image
    cache[('stack_thumbnail_video', stack.id)] = thumbnail_video
