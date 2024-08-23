import numpy as np

from geometry.projection import Projection


class MercatorProjection(Projection):

  def __init__(self, lon_0=0):
    self.lon_0 = lon_0

  def project(self, lon, lat):
    x = lon - self.lon_0
    y = np.log(np.tan((np.pi / 4) + (lat / 2)))

    return x, y

  def unproject(self, x, y):
    lon = self.lon_0 + x
    lat = 2 * np.arctan(np.exp(y)) - (np.pi / 2)

    return lon, lat

  def __repr__(self):
    return f'MercatorProjection(lon_0={self.lon_0})'
