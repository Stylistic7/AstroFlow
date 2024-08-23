import numpy as np

from geometry.projection import Projection
from geometry.projections.galactic import equatorial_to_galactic


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

  def __repr__(self):
    return f'MollweideProjection(lon_0={self.lon_0} R={self.R})'


class MollweideGalacticProjection(Projection):

  def __init__(self, lon_0=0, R=45 * np.sqrt(2)):
    self.proj = MollweideProjection(lon_0, R)

  def project(self, lon, lat):
    theta, phi = equatorial_to_galactic(lon, lat)
    return self.proj.project(theta, phi)

  def unproject(self, x, y):
    raise NotImplementedError

  def __repr__(self):
    return f'MollweideGalacticProjection(lon_0={self.lon_0} R={self.R})'
