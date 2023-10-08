import time
import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon
import uuid
import hashlib

from cache import cache
import file_io
import plate_solving
import spatial_association

import matplotlib.pyplot as plt


class FileList(list):
  def __init__(self, *args):
    if args:
      list.__init__(self, sorted(*args))


def hash_file_list(x):
  if x == None:
    return None
  return hashlib.sha256(bytes(str(sorted(x)), 'utf-8')).hexdigest()


def _min_footprint(footprints):
  min_ra = np.min([p[0] for f in footprints for p in f])
  max_ra = np.max([p[0] for f in footprints for p in f])
  min_dec = np.min([p[1] for f in footprints for p in f])
  max_dec = np.max([p[1] for f in footprints for p in f])

  return [[min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec],
          [min_ra, max_dec]]


# def _get_plate_solutions(light_frames):
#   return {
#       k: v
#       for k, v in cache['plate_solutions'].items() if k in light_frames
#   }


@dataclass
class Stack:
  # id: uuid.UUID = field(default_factory=lambda: str(uuid.uuid4()))
  id: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  enable: bool = False

  target: str = None
  rig: str = None
  filter: str = None

  camera_configs = None

  light_frames: FileList = field(default_factory=FileList)
  dark_frames: FileList = field(default_factory=FileList)
  bias_frames: FileList = field(default_factory=FileList)
  flat_frames: FileList = field(default_factory=FileList)

  flat_frame_associations: list = field(default_factory=list)

  exposure_time_s: int = None

  center: list[float] = field(default_factory=list)
  footprint: list[tuple[float]] = None
  min_footprint: list[tuple[float]] = None
  aspect_ratio: float = None

  best_light_frame: str = None

  master: str = None
  post_processed: str = None

  def store_metadata(self, fits_headers, plate_solutions):
    self.store_id()
    self.store_exposure_time_s(fits_headers)
    self.store_center(plate_solutions)
    self.store_footprint(plate_solutions)
    self.store_min_footprint(plate_solutions)
    self.store_aspect_ratio()

  # TODO: Make a Stack factory function that either finds the same stack
  # alrady in the cache, or makes a new one, computing all its metadata.
  # TODO: Needs to include the flat-frame associations to be complete.
  # def store_id(self):
  #   self.id = (f'{self.target}_{self.rig}_{self.filter}')

  def hash_flat_frame_associations(self):
    hashes = ''
    for light_frames, flat_frames in sorted(self.flat_frame_associations):
      hashes += hash_file_list(light_frames) + '_' + hash_file_list(flat_frames)

    return hashlib.sha256(bytes(hashes, 'utf-8')).hexdigest()

  def store_id(self):
    self.id = (f'stack_{hash_file_list(self.light_frames)}_'
               f'{hash_file_list(self.dark_frames)}_'
               f'{hash_file_list(self.bias_frames)}_'
               f'{hash_file_list(self.flat_frames)}_'
               f'{self.hash_flat_frame_associations()}')
    print(f'### stored stack id {self.id}')

  def footprint_polygon(self):
    return Polygon(self.footprint)

  def store_center(self, plate_solutions, plot=False):
    # print('### Entering store_center()')
    # for f in self.light_frames:
    #   print('  light frame', f)
    # plate_solutions = _get_plate_solutions(self.light_frames)
    centers = [
        p.center for f, p in plate_solutions.items() if f in self.light_frames
    ]
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

    self.center = list(mean_center)

  def store_footprint(self, plate_solutions):
    # This might be too Draconian, because it includes outliers.
    # TODO: Use RANSAC to compute a better footprint.
    # plate_solutions = _get_plate_solutions(self.light_frames)
    footprints = [
        Polygon(p.footprint) for f, p in plate_solutions.items()
        if f in self.light_frames
    ]
    footprint = functools.reduce(Polygon.intersection, footprints,
                                 footprints[0])
    self.footprint = list(footprint.exterior.coords)
    # print('self.footprint', self.footprint)

  def store_min_footprint(self, plate_solutions):
    # plate_solutions = _get_plate_solutions(self.light_frames)
    footprints = [
        p.footprint for f, p in plate_solutions.items()
        if f in self.light_frames
    ]
    self.min_footprint = _min_footprint(footprints)

  def store_aspect_ratio(self):
    min_x = np.min([p[0] for p in self.min_footprint])
    max_x = np.max([p[0] for p in self.min_footprint])
    min_y = np.min([p[1] for p in self.min_footprint])
    max_y = np.max([p[1] for p in self.min_footprint])
    self.aspect_ratio = (max_x - min_x) / (max_y - min_y)

  def store_exposure_time_s(self, fits_headers):
    self.exposure_time_s = sum(
        fits_headers[f]['EXPOSURE'] for f in self.light_frames)

  def __str__(self):
    return (
        f'Stack(target={self.target}, status={self.status}, rig={self.rig}, '
        f'filter={self.filter}, {len(self.light_frames)} light frames)')

  def __eq__(self, other):
    return (self.light_frames == other.light_frames and
            self.dark_frames == other.dark_frames and
            self.bias_frames == other.bias_frames and
            self.flat_frames == other.flat_frames and
            self.camera_configs == other.camera_configs and
            self.flat_frame_associations == other.flat_frame_associations)
