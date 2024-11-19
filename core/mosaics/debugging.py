import matplotlib.pyplot as plt
import numpy as np

from context import context
from mosaics import identification
from targets.types import TargetHash
from geometry.projections.mercator import MercatorProjection


def plot_mosaic(mosaic: list[TargetHash], title: str = 'Mosaic') -> None:
  targets = context.grouped_targets

  plt.figure()
  plt.title(title)

  for i, target_id in enumerate(mosaic):
    target = targets[target_id]

    color = 'black'

    projection = MercatorProjection()
    ra, dec = (np.radians(np.array(x)) for x in target.center)
    center = projection.project(ra, dec)

    print(f'### center is {center}')
    print(f'### mercator is', target.projected_footprints['mercator']['coords'])

    overlapping_targets = [
        tid for tid in mosaic if identification.targets_compatible_overlap(
            targets[target_id], targets[tid]) !=
        identification.Overlap.INCOMPATIBLE
    ]
    print(
        f'### overlapping_targets for target {target_id} are {overlapping_targets}'
    )

    x, y = zip(*target.projected_footprints['mercator']['coords'])
    plt.plot(x, y, color=color)
    plt.text(center[0],
             center[1],
             f'{i}: {target.stable_hash}, {len(overlapping_targets)}',
             fontsize=12,
             color='blue')
    plt.scatter([center[0]], [center[1]], color=color)

  plt.show()
