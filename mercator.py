import numpy as np


class MercatorProjection:
  def __init__(self, min_lon, max_lon, width_pixels):
    self.width_pixels = width_pixels
    self.R = self.width_pixels / (max_lon - min_lon)
    self.min_lon = min_lon
    self.max_lon = max_lon
    self.middle_lon = (max_lon + min_lon) / 2

  def project(self, lon, lat):
    x = self.R * (lon - self.min_lon)
    y = (self.width_pixels /
         2) + self.R * np.log(np.tan((np.pi / 4) + (lat / 2)))

    return x, y

  def unproject(self, x, y):
    lon = self.min_lon + x / self.R
    lat = 2 * np.arctan(np.exp(y / self.R)) - np.pi / 2

    return lon, lat