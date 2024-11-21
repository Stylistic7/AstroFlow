import numpy as np
from spherical_geometry import great_circle_arc, vector
from spherical_geometry.polygon import SphericalPolygon  # type: ignore

from geometry.projection import Projection


def project_polygon(polygon: SphericalPolygon,
                    projection: Projection) -> tuple[list[float], list[float]]:
  xs, ys = [], []
  points = list(polygon.points)
  for A, B in zip(points[0:-1], points[1:]):
    length = np.rad2deg(great_circle_arc.length(A, B))
    if not np.isfinite(length):
      length = 2
    interpolated = great_circle_arc.interpolate(A, B, length * 4)
    ra, dec = vector.vector_to_radec(interpolated[:, 0],
                                     interpolated[:, 1],
                                     interpolated[:, 2],
                                     degrees=False)

    x, y = projection.project(ra, dec)
    xs.extend(x)
    ys.extend(y)

  return xs, ys
