import numpy as np
from astropy.coordinates import SkyCoord

from projection import Projection


def equatorial_to_galactic(ra, dec):
  eq = SkyCoord(ra, dec, unit='rad')

  theta = np.radians((180 - eq.galactic.l.value) % 360 - 180)
  phi = np.radians(eq.galactic.b.value)

  return theta, phi


# TODO: This is akwardly-named, because it isn't a 3D to 2D projection at all.
class GalacticProjection(Projection):

  def project(self, lon, lat):
    return equatorial_to_galactic(lon, lat)

  def unproject(self, x, y):
    raise NotImplementedError
