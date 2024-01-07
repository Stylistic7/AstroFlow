import numpy as np
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon

from projection import compute_viewbox
import mercator
import sinusoidal
import mollweide


def project_footprint(polygon, projection):
  ra, dec = next(polygon.to_radec())  # degrees
  ra, dec = (np.radians(np.array(x)) for x in (ra, dec))
  x, y = projection.project(ra, dec)
  return Polygon(zip(x, y))


def all_projections(polygon):
  projections = {
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