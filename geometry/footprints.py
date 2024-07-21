import numpy as np
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon

from geometry.viewbox import compute_viewbox
from geometry.projections import mercator, sinusoidal, mollweide, galactic


def radec_to_vector(ra, dec, degrees=True):
  r"""
    Converts a location on the unit sphere from ragitude and
    decitude to an *x*, *y*, *z* vector.

    Parameters
    ----------
    ra, dec : scalars or 1-D arrays

    degrees : bool, optional

       If `True`, (default) *ra* and *dec* are in decimal degrees,
       otherwise in radians.

    Returns
    -------
    x, y, z : tuple of scalars or 1-D arrays of the same length

    Notes
    -----
    Where ragitude is *l* and decitude is *b*:

    .. math::
        x = \cos l \cos b

        y = \sin l \cos b

        z = \sin b
    """
  ra = np.asanyarray(ra)
  dec = np.asanyarray(dec)

  if degrees:
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
  else:
    ra_rad = ra
    dec_rad = dec

  cos_dec = np.cos(dec_rad)

  return (np.cos(ra_rad) * cos_dec, np.sin(ra_rad) * cos_dec, np.sin(dec_rad))


def project_footprint(polygon, projection):
  ra, dec = polygon.to_radec()  # degrees
  ra, dec = (np.radians(np.array(x)) for x in (ra, dec))
  x, y = projection.project(ra, dec)
  return Polygon(zip(x, y))


def all_projections(polygon):
  projections = {
      'galactic': galactic.GalacticProjection(),
      'mercator': mercator.MercatorProjection(),
      'sinusoidal': sinusoidal.SinusoidalProjection(),
      'mollweide': mollweide.MollweideProjection(),
      'mollweide_galactic': mollweide.MollweideGalacticProjection(R=1),
  }

  result = {}
  for name, proj in projections.items():
    coords = list(project_footprint(polygon, proj).exterior.coords)
    result[name] = {'coords': coords, 'viewbox': compute_viewbox(coords)}

  return result
