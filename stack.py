import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon

import plate_solving
import spatial_association

import matplotlib.pyplot as plt


@dataclass
class Stack:
  target: str = None
  rig: str = None
  filter: str = None

  light_frames: list[str] = field(default_factory=list)
  dark_frames: list[str] = field(default_factory=list)
  bias_frames: list[str] = field(default_factory=list)
  flat_frames: list[str] = field(default_factory=list)

  flat_frame_associations: list = field(default_factory=list)

  camera_configs = None

  center: list[float] = field(default_factory=list)
  footprint: list[tuple[float]] = None

  master: str = None
  post_processed: str = None

  def compute_center(self, plot=False):
    print('### Entering compute_center()')
    for f in self.light_frames:
      print('  light frame', f)
    plate_solutions = plate_solving.solve_fields(self.light_frames)
    centers = [p.center for p in plate_solutions.values()]
    print('centers', centers)
    mean_center = np.mean(centers, axis=0)
    print('  mean_center:', mean_center)
    std_dev_center = np.std(centers, axis=0)
    print('  std_dev_center:', std_dev_center)

    if plot:
      plt.figure()
      plt.scatter([c[0] for c in centers], [c[1] for c in centers])
      plt.title(f'{self.target} std. dev. {std_dev_center}')
      plt.show()

    return list(mean_center)

  def compute_footprint(self):
    # This might be too Draconian, because it includes outliers.
    # TODO: Use RANSAC to compute a better footprint.
    plate_solutions = plate_solving.solve_fields(self.light_frames)
    footprints = [Polygon(f) for _, f, _ in plate_solutions.values()]
    footprint = functools.reduce(Polygon.intersection, footprints,
                                 footprints[0])
    return list(footprint.exterior.coords)

  def store_center_and_footprint(self):
    self.center = self.compute_center()
    self.footprint = self.compute_footprint()

  def __repr__(self):
    return (
        f'Stack(target={self.target}, rig={self.rig}, filter={self.filter}, '
        f'{len(self.light_frames)} light frames)')