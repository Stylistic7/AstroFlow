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

import file_io

from cache import cache


def resize_array(array, new_width):
  height, width = array.shape
  new_height = (height / width) * new_width
  # print(f'resize_array() from height {height} width {width} '
  #       f'to height {new_height} width {new_width}')

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


def thumbnail(filename, rotate, width=640):
  if filename.endswith('.xisf'):
    im = file_io.read_xisf_image(filename)
  elif filename.endswith('.fit') or filename.endswith('.fits'):
    im = file_io.read_fits_image(filename)
    if rotate:
      im = np.rot90(im, 2)

  return autostretch(resize_array(im, width))


# TODO: Pass width down through function hierarchy.
# TODO: Consider thumbnailing every image and storing them in the cache
# independently. If a mosaic config or threshold changes a stack's composition,
# the blink video could then be regenerated very quickly. This is also an
# embarassingly-parallel operation, as is the video encoding.
def thumbnails(filenames, plate_solutions, width=640):
  filenames_and_rotation_angles = [(f, p.rotation_angle)
                                   for f, p in plate_solutions.items()
                                   if f in filenames]
  with Pool() as p:
    return p.starmap(thumbnail, filenames_and_rotation_angles)


def stack_thumbnail_image_and_video(stack, width=640):
  height, width = resize_array(file_io.read_fits_image(stack.light_frames[0]),
                               width).shape
  height = int(((height + 8 - 1) / 8) * 8)
  print(f'Creating video of width {width}, height {height} '
        f'with {len(stack.light_frames)} frames')

  with tempfile.TemporaryDirectory() as tmp_dir:
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video_filename = os.path.join(tmp_dir, 'output.webm')
    out = cv2.VideoWriter(video_filename, fourcc, 10.0, (width, height), False)

    thumbnails = [cache[('thumbnail_image', f)] for f in stack.light_frames]
    index = np.argmax(map(np.std, thumbnails))
    thumbnail_image = array_to_png(thumbnails[index])

    for im in thumbnails:
      im = (im * 255.0).astype(np.uint8)
      frame = np.zeros((height, width), dtype=np.uint8)
      frame[0:im.shape[0], 0:im.shape[1]] = im
      # print('Writing frame to video')
      out.write(frame)

    out.release()
    thumbnail_video = open(video_filename, 'rb').read()

    return thumbnail_image, thumbnail_video


# if __name__ == '__main__':
# frames = file_io.find_files('/Users/nobody/Projects/Astro/Raw/M101', '*.fit')
# stack = Stack()
# stack.light_frames = [
#     f for f in frames if file_io.fits_matches(f, 'IMAGETYP', 'Light')
# ]
# print(f'First light frame is {stack.light_frames[0]}')
# stack.light_frames = list(sorted(stack.light_frames))
# video = thumbnail_video(stack)

# video_filename = '/tmp/foo.webm'
# open(video_filename, 'wb').write(video)
# print(f'Wrote video to {video_filename}')