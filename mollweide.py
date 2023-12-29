import numpy as np
from astropy.coordinates import SkyCoord

from projection import Projection


def equatorial_to_galactic(ra, dec):
  eq = SkyCoord(ra, dec, unit='deg')

  theta = np.radians((180 - eq.galactic.l.value) % 360 - 180)
  phi = np.radians(eq.galactic.b.value)

  return theta, phi


class MollweideProjection(Projection):
  def __init__(self, lon_0=0, R=45 * np.sqrt(2)):
    self.lon_0 = lon_0
    self.R = R

  def _solve_theta(self, phi):
    theta = 0
    for i in range(10):
      theta = theta - (2 * theta + np.sin(2 * theta) -
                       np.pi * np.sin(phi)) / (4 * np.cos(theta)**2.0)

    return theta

  def project(self, lon, lat):
    theta = self._solve_theta(lat)

    x = self.R * (2 * np.sqrt(2) / np.pi) * (lon - self.lon_0) * np.cos(theta)
    y = self.R * np.sqrt(2) * np.sin(theta)

    return x, y

  def unproject(self, x, y):
    theta = np.arcsin(y / (self.R * np.sqrt(2)))
    lon = self.lon_0 + (x * np.pi) / (2 * self.R * np.sqrt(2) * np.cos(theta))
    lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)

    lon = np.where((lon >= 0) & (lon <= 2 * np.pi), lon, np.nan)
    lat = np.where((lat >= -np.pi) & (lat <= np.pi), lat, np.nan)

    return lon, lat