import numpy as np

from geometry.projection import Projection


class SinusoidalProjection(Projection):

  def __init__(self, lon_0=0):
    self.lon_0 = lon_0

  def project(self, lon, lat):
    x = (lon - self.lon_0) * np.cos(lat)
    y = lat

    return x, y

  def unproject(self, x, y):
    lon = x / np.cos(y) + self.lon_0
    lat = y

    lon = np.where((lon >= 0) & (lon <= 2 * np.pi), lon, np.nan)
    lat = np.where((lat >= -np.pi) & (lat <= np.pi), lat, np.nan)

    return lon, lat
