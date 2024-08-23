import numpy as np


def largest_interior_rectangle(image):

  def contains_nan(x):
    return np.isnan(np.min(x))

  def find_seed_point():
    height, width = image.shape
    while True:
      y, x = np.random.randint(height), np.random.randint(width)
      if not contains_nan(image[y - 1:y + 1, x - 1:x + 1]):
        return y, x

  def try_rectangle(top, left, bottom, right):
    height, width = image.shape
    if top < 0 or left < 0 or right > width or bottom > height:
      return False

    if contains_nan(image[top:bottom, left:left + 1]):  # left edge
      return False
    if contains_nan(image[top:bottom, right - 1:right]):  # right edge
      return False
    if contains_nan(image[top:top + 1, left:right]):  # top edge
      return False
    if contains_nan(image[bottom - 1:bottom, left:right]):  # bottom edge
      return False

    return True

  y, x = find_seed_point()
  top, left = y - 1, x - 1
  bottom, right = y + 1, x + 1

  while True:
    changed = False

    if try_rectangle(top - 1, left, bottom, right):
      changed = True
      top -= 1
    if try_rectangle(top, left - 1, bottom, right):
      changed = True
      left -= 1
    if try_rectangle(top, left, bottom + 1, right):
      changed = True
      bottom += 1
    if try_rectangle(top, left, bottom, right + 1):
      changed = True
      right += 1

    if not changed:
      break

  return top, left, bottom, right
