import matplotlib.pyplot as plt
import numpy as np
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from spherical_geometry import great_circle_arc, vector

from context import context
from mosaics import identification, Mosaic
from geometry.projection import Projection
from targets.types import TargetHash
from geometry.projections.mercator import MercatorProjection
from geometry.projections.sinusoidal import SinusoidalProjection
from geometry.projections.mollweide import MollweideProjection


def _draw_polygon(p: SphericalPolygon,
                  projection: Projection,
                  color: str = 'black') -> None:
  points = list(p.points)

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
    plt.plot(x, y, color=color)


def plot_mosaic(mosaic: Mosaic, title: str = 'Mosaic') -> None:
  targets = context.grouped_targets

  plt.figure()
  plt.title(title)

  for i, target_id in enumerate(mosaic.target_ids):
    target = targets[target_id]
    tile_indices = mosaic.tile_indices[i]

    projection_name = 'mercator'
    projection = MercatorProjection()

    ra, dec = (np.radians(np.array(x)) for x in target.center)
    center = projection.project(ra, dec)

    print(f'### center is {center}')
    print(f'### mercator is',
          target.projected_footprints[projection_name]['coords'])

    overlapping_targets = [
        tid for tid in mosaic.target_ids
        if identification.targets_compatible_overlap(targets[target_id],
                                                     targets[tid]) !=
        identification.Overlap.INCOMPATIBLE
    ]
    print(
        f'### overlapping_targets for target {target_id} are {overlapping_targets}'
    )

    _draw_polygon(target.footprint, projection, color='blue')
    plt.text(
        center[0],
        center[1],
        f'{tile_indices[0]}, {tile_indices[1]}: {len(overlapping_targets)}',
        fontsize=12,
        color='blue')
    plt.scatter([center[0]], [center[1]], color='black')

  _draw_polygon(mosaic.footprint, projection, color='red')

  plt.show()
