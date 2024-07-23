import numpy as np
import unittest

from mosaics.largest_interior_rectangle import largest_interior_rectangle


class TestLargestInteriorRectangle(unittest.TestCase):

  def test_central_square(self):
    array = np.ones((200, 100)) * np.nan
    array[20:180, 20:80] = 1

    top, left, bottom, right = largest_interior_rectangle(array)

    self.assertEqual(top, 20)
    self.assertEqual(left, 20)
    self.assertEqual(bottom, 180)
    self.assertEqual(right, 80)


if __name__ == "__main__":
  unittest.main()
