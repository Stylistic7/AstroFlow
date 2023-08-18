from PIL import Image
from xisf import XISF
import numpy as np
from numpy import abs, minimum, maximum, median

# TODO: Parameterize this.
PROCESSED_DIR = '/Users/nobody/Projects/Astro/Processed'


def read_xisf(filename):
  xisf = XISF(filename)
  return xisf.read_image(0).squeeze()


def resize_array(array):
  height, width = array.shape
  print('original shape', array.shape)
  image = Image.fromarray(array)
  image.resize((height / 4, width / 4), PIL.Resampling.BILINEAR)
  array = np.asarray(image)
  print('resized shape', array.shape)
  return array


def autostretch(image):
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


def array_to_png(array, filename):
  array = (array * 255.0).astype(np.uint8)

  print('array.shape', array.shape, 'array.dtype', array.dtype)

  im = Image.fromarray(array)
  im.save('/tmp/foo.png')


if __name__ == '__main__':
  filename = "/Users/nobody/Projects/Astro/Processed/Network nebula_RA_313.6_DEC_30.1/masters/master_light_Network nebula_RA_313.6_DEC_30.1_C11_H_v4.xisf"

  array_to_png(autostretch(resize_array(read_xisf(filename))), '/tmp/foo.png')
