import numpy as np
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon


def project_footprint(polygon, projection):
  ra, dec = next(polygon.to_radec())
  ra, dec = (np.radians(np.array(x)) for x in (ra, dec))
  x, y = projection.project(ra, dec)
  return Polygon(zip(x, y))
