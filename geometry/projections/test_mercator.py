import unittest
import numpy as np

from .mercator import MercatorProjection


class TestMercatorProjection(unittest.TestCase):

  def setUp(self):
    self.proj = MercatorProjection()

  def test_meridian(self):
    x, y = self.proj.project(np.pi, 0)
    self.assertTrue(np.allclose((x, y), (np.pi, 0), atol=0.05))
    lon, lat = self.proj.unproject(x, y)
    self.assertTrue(np.allclose((lon, lat), (np.pi, 0)))

    x, y = self.proj.project(np.pi, np.radians(85))
    self.assertTrue(np.allclose((x, y), (np.pi, np.pi), atol=0.05))
    lon, lat = self.proj.unproject(x, y)
    self.assertTrue(np.allclose((lon, lat), (np.pi, np.radians(85))))

    x, y = self.proj.project(np.pi, np.radians(-85))
    self.assertTrue(np.allclose((x, y), (np.pi, -np.pi), atol=0.05))
    lon, lat = self.proj.unproject(x, y)
    self.assertTrue(np.allclose((lon, lat), (np.pi, np.radians(-85))))

  def test_equator(self):
    x, y = self.proj.project(0, 0)
    self.assertTrue(np.allclose((x, y), (0, 0), atol=0.05))
    lon, lat = self.proj.unproject(x, y)
    self.assertTrue(np.allclose((lon, lat), (0, 0)))

    x, y = self.proj.project(2 * np.pi, 0)
    self.assertTrue(np.allclose((x, y), (2 * np.pi, 0), atol=0.05))
    lon, lat = self.proj.unproject(x, y)
    self.assertTrue(np.allclose((lon, lat), (2 * np.pi, 0)))


if __name__ == "__main__":
  unittest.main()
