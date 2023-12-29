import numpy as np
from shapely.geometry import Polygon

import projection
import mollweide
import mercator
import sinusoidal


# TODO: Use the centroid of the shape to define `lon_0`.
def compute_footprints(items, reduce_fn=Polygon.union):
  projections = {
      'mollweide': mollweide.MollweideProjection(lon_0=np.pi),
      'mercator': mercator.MercatorProjection(lon_0=np.pi),
      'sinusoidal': sinusoidal.SinusoidalProjection(lon_0=np.pi)
  }

  footprints = {}
  for name, proj in projections.items():
    footprint = Polygon()
    for item in items:
      coords = list(item.footprint_polygon().exterior.coords)
      lon, lat = zip(*coords)
      lon, lat = np.radians(np.array(lon)), np.radians(np.array(lat))
      xx, yy = proj.project(lon, lat)
      target_footprint = Polygon(zip(xx, yy))
      footprint = reduce_fn(footprint, target_footprint)

    coords = list(footprint.exterior.coords)
    footprints[name] = dict(coords=coords,
                            viewbox=projection.compute_viewbox(coords))

  return footprints