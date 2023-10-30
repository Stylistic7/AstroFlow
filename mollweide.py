import numpy as np
from astropy.coordinates import SkyCoord


def _solve_theta(phi):
  theta = 0
  for i in range(10):
    theta = theta - (2 * theta + np.sin(2 * theta) -
                     np.pi * np.sin(phi)) / (4 * np.cos(theta)**2.0)

  return theta


def project_Mollweide(lon, lat, R=45 * np.sqrt(2)):
  lon_0 = 0
  theta = _solve_theta(lat)

  print('compare left', 2 * theta + np.sin(2 * theta), 'right',
        np.pi * np.sin(lat))

  x = R * (2 * np.sqrt(2) / np.pi) * (lon - lon_0) * np.cos(theta)
  y = R * np.sqrt(2) * np.sin(theta)

  return x, y


def equatorial_to_galactic(ra, dec):
  eq = SkyCoord(ra, dec, unit='deg')

  theta = np.radians((180 - eq.galactic.l.value) % 360 - 180)
  phi = np.radians(eq.galactic.b.value)

  return theta, phi