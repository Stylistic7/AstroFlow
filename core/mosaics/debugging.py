import matplotlib.pyplot as plt
import numpy as np

from context import context
from mosaics.ransac import RansacResult
from targets.types import TargetHash
from geometry.projections.mercator import MercatorProjection


def plot_ransac_result(mosaic: list[TargetHash],
                       result: RansacResult,
                       plot_title: str = 'RANSAC Result') -> None:
  targets = context.grouped_targets

  plt.figure()
  plt.title(plot_title)

  print('### basis_vectors', result.basis_vectors)

  # Draw the basis vectors.
  for i, basis_vector in enumerate(result.basis_vectors.T):
    plt.plot([result.origin[0], result.origin[0] + basis_vector[0]],
             [result.origin[1], result.origin[1] + basis_vector[1]],
             label=f"Basis vector {i}",
             linewidth=5,
             color=f"C{i + 2}")

  # Plot the centers of the N x M grid.
  # points = []
  # for x in range(np.max(integral_centers, axis=0)):
  #   for y in range(np.max(integral_centers, axis=1)):
  #     points.append(origin + x * basis_vectors[:, 0] +
  #                   y * basis_vectors[:, 1])
  # xs, ys = [p[0] for p in points], [p[1] for p in points]
  # plt.scatter(xs, ys, color='red')

  for target_id in mosaic:
    target = targets[target_id]

    color = 'black' if target_id in result.target_ids else '#dddddd'

    projection = MercatorProjection()
    ra, dec = (np.radians(np.array(x)) for x in target.center)
    center = projection.project(ra, dec)

    print(f'### center is {center}')
    print(f'### mercator is', target.projected_footprint['mercator']['coords'])

    x, y = zip(*target.projected_footprint['mercator']['coords'])
    plt.plot(x, y, color=color)
    plt.scatter([center[0]], [center[1]], color=color)

  plt.show()
