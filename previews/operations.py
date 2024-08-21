import numpy as np
from numpy import abs, minimum, maximum, median
from skimage.transform import resize
import cv2


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
