import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon

from cache import cache
import file_io
import plate_solving
import spatial_association

import matplotlib.pyplot as plt


def _min_footprint(footprints):
  min_ra = np.min([p[0] for f in footprints for p in f])
  max_ra = np.max([p[0] for f in footprints for p in f])
  min_dec = np.min([p[1] for f in footprints for p in f])
  max_dec = np.max([p[1] for f in footprints for p in f])

  return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
          [min_ra, max_dec]]


def _get_plate_solutions(light_frames):
  return {
      k: v
      for k, v in cache['plate_solutions'].items() if k in light_frames
  }


@dataclass
class Stack:
  id: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  target: str = None
  rig: str = None
  filter: str = None

  camera_configs = None

  light_frames: list[str] = field(default_factory=list)
  dark_frames: list[str] = field(default_factory=list)
  bias_frames: list[str] = field(default_factory=list)
  flat_frames: list[str] = field(default_factory=list)

  flat_frame_associations: list = field(default_factory=list)

  exposure_time_s: int = None

  thumbnail_image_url: str = None
  thumbnail_video_url: str = None

  center: list[float] = field(default_factory=list)
  footprint: list[tuple[float]] = None
  min_footprint: list[tuple[float]] = None
  aspect_ratio: float = None

  master: str = None
  post_processed: str = None

  def store_metadata(self):
    self.id = self.compute_id()
    self.exposure_time_s = self.compute_exposure_time_s()
    self.center = self.compute_center()
    self.footprint = self.compute_footprint()
    self.min_footprint = self.compute_min_footprint()
    self.aspect_ratio = self.compute_aspect_ratio()

  # TODO: Make a Stack factory function that either finds the same stack
  # alrady in the cache, or makes a new one, computing all its metadata.
  def compute_id(self):
    return (f'{self.target}_{self.rig}_{self.filter}_'
            f'{self.exposure_time_s}_{self.status}_'
            f'{hash(frozenset(self.camera_configs))}_'
            f'{hash(frozenset(self.light_frames))}_'
            f'{hash(frozenset(self.dark_frames))}_'
            f'{hash(frozenset(self.bias_frames))}_'
            f'{hash(frozenset(self.flat_frames))}_'
            f'{hash(frozenset(self.flat_frame_associations))}')

  def footprint_polygon(self):
    return Polygon(self.footprint)

  def compute_center(self, plot=False):
    # print('### Entering compute_center()')
    # for f in self.light_frames:
    #   print('  light frame', f)
    plate_solutions = _get_plate_solutions(self.light_frames)
    centers = [p.center for p in plate_solutions.values()]
    # print('centers', centers)
    mean_center = np.mean(centers, axis=0)
    # print('  mean_center:', mean_center)
    std_dev_center = np.std(centers, axis=0)
    # print('  std_dev_center:', std_dev_center)

    if plot:
      plt.figure()
      plt.scatter([c[0] for c in centers], [c[1] for c in centers])
      plt.title(f'{self.target} std. dev. {std_dev_center}')
      plt.show()

    return list(mean_center)

  def compute_footprint(self):
    # This might be too Draconian, because it includes outliers.
    # TODO: Use RANSAC to compute a better footprint.
    plate_solutions = _get_plate_solutions(self.light_frames)
    footprints = [Polygon(p.footprint) for p in plate_solutions.values()]
    footprint = functools.reduce(Polygon.union, footprints, footprints[0])
    return list(footprint.exterior.coords)

  def compute_min_footprint(self):
    plate_solutions = _get_plate_solutions(self.light_frames)
    footprints = [f.footprint for f in plate_solutions.values()]
    return _min_footprint(footprints)

  def compute_aspect_ratio(self):
    min_x = np.min([p[0] for p in self.min_footprint])
    max_x = np.max([p[0] for p in self.min_footprint])
    min_y = np.min([p[1] for p in self.min_footprint])
    max_y = np.max([p[1] for p in self.min_footprint])
    return (max_x - min_x) / (max_y - min_y)

  def compute_exposure_time_s(self):
    return sum(file_io.read_fits(f)['EXPOSURE'] for f in self.light_frames)

  def __repr__(self):
    return (
        f'Stack(target={self.target}, status={self.status}, rig={self.rig}, '
        f'filter={self.filter}, {len(self.light_frames)} light frames)')

  def __eq__(self, other):
    return (
        list(sorted(self.light_frames)) == list(sorted(other.light_frames)) and
        list(sorted(self.dark_frames)) == list(sorted(other.dark_frames)) and
        list(sorted(self.bias_frames)) == list(sorted(other.bias_frames)) and
        list(sorted(self.flat_frames)) == list(sorted(other.flat_frames)) and
        list(sorted(self.camera_configs)) == list(sorted(other.camera_configs))
        and list(sorted(self.flat_frame_associations)) == list(
            sorted(other.flat_frame_associations)))
