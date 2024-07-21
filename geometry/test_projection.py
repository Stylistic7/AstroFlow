import unittest
import numpy as np
import matplotlib.pyplot as plt

from geometry.projections import mercator
from geometry.projections import mollweide
from geometry.projections import sinusoidal


class TestProjection(unittest.TestCase):

  def test_mercator_whole_sphere(self, debug=False):
    proj = mercator.MercatorProjection(lon_0=0)

    # The Mercator projection projects the entire sphere to the xy plane bounded
    # by x in [0, 2 pi), and y in [-pi, pi].
    viewbox = 0, -np.pi, 2 * np.pi, 2 * np.pi
    lon, lat = proj.sample_viewbox_lon_lat(viewbox, 1000)

    self.assertEqual(np.min(lon), 0)
    self.assertEqual(np.max(lon), 2 * np.pi)

    self.assertTrue(np.allclose(np.min(lat), np.radians(-85), atol=0.05))
    self.assertTrue(np.allclose(np.max(lat), np.radians(85), atol=0.05))

    if debug:
      plt.figure()
      plt.imshow(lon)
      plt.title('lon')

      plt.figure()
      plt.imshow(lat)
      plt.title('lat')

      plt.show()

  def test_sinusoidal_whole_sphere(self, debug=False):
    proj = sinusoidal.SinusoidalProjection(lon_0=np.pi)

    # The sinusoidal projection projects the entire sphere to the xy plane,
    # bounded by x in [0, 2 pi), and y in [-pi, pi].
    viewbox = -np.pi, -np.pi / 2, 2 * np.pi, np.pi
    lon, lat = proj.sample_viewbox_lon_lat(viewbox, 1000)

    self.assertEqual(np.min(lat), -np.pi / 2)
    self.assertEqual(np.max(lat), np.pi / 2)

    if debug:
      plt.figure()
      plt.imshow(lon)
      plt.title('lon')

      plt.figure()
      plt.imshow(lat)
      plt.title('lat')

      plt.show()

  def test_mollweide_whole_sphere(self, debug=False):
    R = 1
    proj = mollweide.MollweideProjection(R=R, lon_0=np.pi)

    # The Mollweide projection projects the entire sphere to the xy plane
    # bounded by x in [-2 R sqrt(2), 2 R sqrt(2)] and y in
    # [-R sqrt(2), R sqrt(2)].
    viewbox = -2 * R * np.sqrt(2), -R * np.sqrt(2), 4 * R * np.sqrt(
        2), 2 * R * np.sqrt(2)
    lon, lat = proj.sample_viewbox_lon_lat(viewbox, 1000)

    if debug:
      plt.figure()
      plt.imshow(lon)
      plt.title('lon')

      plt.figure()
      plt.imshow(lat)
      plt.title('lat')

      plt.show()


if __name__ == "__main__":
  unittest.main()
