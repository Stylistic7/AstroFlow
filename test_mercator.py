import unittest
import numpy as np

import mercator


class TestMercatorProjection(unittest.TestCase):
  def setUp(self):
    # The image should be 1000 x 1000 pixels, including latitudes in the range
    # [-85, 85].
    self.proj = mercator.MercatorProjection(0, 2 * np.pi, 1000)

  def test_meridian(self):
    x, y = self.proj.project(np.pi, 0)
    self.assertTrue(np.allclose((x, y), (500, 500), atol=0.01))

    x, y = self.proj.project(np.pi, np.radians(85))
    self.assertTrue(np.allclose((x, y), (500, 998.36), atol=0.01))

    x, y = self.proj.project(np.pi, np.radians(-85))
    self.assertTrue(np.allclose((x, y), (500, 1.64), atol=0.01))

  def test_equator(self):
    x, y = self.proj.project(0, 0)
    self.assertTrue(np.allclose((x, y), (0, 500), atol=0.01))

    x, y = self.proj.project(2 * np.pi, 0)
    self.assertTrue(np.allclose((x, y), (1000, 500), atol=0.01))


if __name__ == "__main__":
  unittest.main()