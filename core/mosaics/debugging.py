import matplotlib.pyplot as plt
import numpy as np
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely import Polygon  # type: ignore

from context import context
from mosaics import identification
from mosaics.types import Mosaic
from geometry.projection import Projection
from geometry.polygons import project_polygon
from geometry.projections.mercator import MercatorProjection


def draw_polygon(polygon: Polygon, color: str = 'black') -> None:
  x, y = zip(*polygon.exterior.coords)
  plt.plot(x, y, color=color)


def draw_spherical_polygon(polygon: SphericalPolygon,
                           projection: Projection,
                           color: str = 'black') -> None:
  x, y = project_polygon(polygon, projection)
  plt.plot(x, y, color=color)


def plot_mosaic(mosaic: Mosaic, title: str = 'Mosaic') -> None:
  targets = context.grouped_targets

  plt.figure()
  plt.title(title)

  for i, target_id in enumerate(mosaic.target_ids):
    target = targets[target_id]

    projection_name = 'mercator'
    projection = MercatorProjection()

    ra, dec = (np.radians(np.array(x)) for x in target.center)
    center = projection.project(ra, dec)

    print(f'### center is {center}')
    # print(f'### mercator is',
    #       target.projected_footprints[projection_name]['coords'])

    overlapping_targets = [
        tid for tid in mosaic.target_ids
        if identification.targets_overlap(targets[target_id], targets[tid]) !=
        identification.Overlap.INCOMPATIBLE
    ]
    print(f'### overlapping_targets for target {target_id} '
          f'are {overlapping_targets}')

    draw_spherical_polygon(target.footprint, projection, color='blue')
    plt.text(
        center[0],
        center[1],
        f'{target_id}: {len(overlapping_targets)}',
        fontsize=12,
        color='blue')
    plt.scatter([center[0]], [center[1]], color='black')

  draw_spherical_polygon(mosaic.footprint, projection, color='red')

  plt.show()
