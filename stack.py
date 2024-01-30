import os
import numpy as np
from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon
import astropy
import hashlib
from enum import Enum, unique
import xxhash
import matplotlib.pyplot as plt

import footprints


def ra_to_hms(ra):
  hours, remainder = np.divmod(ra * 3600, 15 * 3600)
  minutes, remainder = np.divmod(remainder, 15 * 60)
  seconds = remainder / 15
  return f'{hours:.0f}h {minutes:.0f}m {seconds:.0f}s'


def dec_to_dms(dec):
  degrees, remainder = np.divmod(dec * 3600, 3600)
  minutes, remainder = np.divmod(remainder, 60)
  seconds = remainder
  return f"{degrees:.0f}° {minutes:.0f}' {seconds:.0f}\""


def hash_string(s):
  return hashlib.sha256(bytes(s, 'utf-8')).hexdigest()


def hash_file_list(x):
  if x == None:
    return None
  return hash_string(str(sorted(x)))


# TODO: This is a mess right now. Design a better set of statuses.
@unique
class Status(str, Enum):
  READY: str = "READY"
  NOT_READY: str = "NOT_READY"

  CANNOT_RUN: str = "CANNOT_RUN"
  NEEDS_TO_RUN: str = "NEEDS_TO_RUN"
  RUNNING: str = "RUNNING"
  SUCCESS: str = "SUCCESS"
  FAILURE: str = "FAILURE"


@dataclass(order=True)
class Stack:
  id: str = None
  content_hash: str = None

  status: Status = Status.CANNOT_RUN

  target: str = None
  image_scale: float = None
  rotation_angle: float = None
  width_pixels: int = None
  height_pixels: int = None
  filter: str = None

  camera_configs = None

  light_frames: list = field(default_factory=list)
  dark_frames: list = field(default_factory=list)
  bias_frames: list = field(default_factory=list)
  flat_frames: list = field(default_factory=list)

  flat_frame_associations: list = field(default_factory=list)

  exposure_time_s: int = None
  footprint: SphericalPolygon = None
  projected_footprint: dict[str, Polygon] = None

  # TODO: Can these just be removed?
  best_light_frame: str = None
  image_scales_not_all_close: bool = False
  rotation_angles_not_all_close: bool = False

  def __post_init__(self):
    self.light_frames = sorted(self.light_frames)
    self.dark_frames = sorted(self.dark_frames)
    self.bias_frames = sorted(self.bias_frames)
    self.flat_frames = sorted(self.flat_frames)

  def normalize(self):
    self.light_frames = sorted(self.light_frames)
    self.dark_frames = sorted(self.dark_frames)
    self.bias_frames = sorted(self.bias_frames)
    self.flat_frames = sorted(self.flat_frames)

  def append_suffix(self, suffix):
    assert self.master
    filename, ext = os.path.splitext(self.master)
    return f'{filename}_{suffix}{ext}'

  def store_metadata(self, fits_headers, plate_solutions):
    self.store_center(plate_solutions)
    self.store_exposure_time_s(fits_headers)
    self.store_image_scale(plate_solutions)
    self.store_rotation_angle(plate_solutions)
    self.store_width_height(fits_headers)
    self.store_id(plate_solutions)
    self.store_content_hash()
    self.store_footprint(plate_solutions)
    self.store_best_light_frame(plate_solutions)
    self.sort_index = self.target

  def store_best_light_frame(self, plate_solutions):
    # TODO: This seems to sometimes think a completely fogged image has a high
    # star count. Maybe look at the sigma setting on the plate solver?
    light_frames_by_stars = sorted(self.light_frames,
                                   key=lambda f: plate_solutions[f].num_stars,
                                   reverse=True)
    self.best_light_frame = light_frames_by_stars[0]

  def store_footprint(self, plate_solutions):
    self.footprint = next(
        SphericalPolygon.from_wcs(
            astropy.io.fits.Header(
                plate_solutions[self.light_frames[0]].header)).points).tolist()
    self.projected_footprint = footprints.all_projections(
        SphericalPolygon(self.footprint))

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)

  def hash_flat_frame_associations(self):
    hashes = ''
    for light_run, flat_run in sorted(self.flat_frame_associations):
      hashes += hash_string(str(light_run)) + '_' + hash_string(str(flat_run))

    return hashlib.sha256(bytes(hashes, 'utf-8')).hexdigest()

  # TODO: Choose a hash length that's maybe more reasonable than the 64-byte
  # hashes used today. When there are only a few dozen or few thousand entities,
  # the 64-byte hash is almost assuredly unnecessarily long.
  def store_content_hash(self):
    self.content_hash = hash_string(
        f'stack_{hash_file_list(self.light_frames)}_'
        f'{hash_file_list(self.dark_frames)}_'
        f'{hash_file_list(self.bias_frames)}_'
        f'{hash_file_list(self.flat_frames)}_'
        f'{self.hash_flat_frame_associations()}')

  def store_id(self, plate_solutions):
    rotation_angle = plate_solutions[self.light_frames[0]].rotation_angle
    string = (f'RA {ra_to_hms(self.center[0])}, '
              f'Dec {dec_to_dms(self.center[1])}, '
              f'Rotation {rotation_angle:.2f}, '
              f'{self.width_pixels}x{self.height_pixels}, '
              f'{self.image_scale:.2f}"/px, '
              f'Filter {self.filter}')
    print('id string', string)
    self.id = hash_string(string)

  def __hash__(self):
    return xxhash.xxh64(self.id).intdigest()

  def store_center(self, plate_solutions, plot=False):
    assert self.light_frames

    centers = [
        p.center for f, p in plate_solutions.items() if f in self.light_frames
    ]
    mean_center = np.mean(centers, axis=0)
    std_dev_center = np.std(centers, axis=0)

    if plot:
      plt.figure()
      plt.scatter([c[0] for c in centers], [c[1] for c in centers])
      plt.title(f'{self.target} std. dev. {std_dev_center}')
      plt.show()

    self.center = list(mean_center)
    assert len(self.center)

  def store_image_scale(self, plate_solutions):
    image_scales = [
        p.image_scale
        for f, p in plate_solutions.items()
        if f in self.light_frames
    ]
    mean = np.mean(image_scales)
    self.image_scale = mean

    self.image_scales_not_all_close = not np.allclose(
        image_scales, mean, rtol=0.02, atol=0)

  def store_rotation_angle(self, plate_solutions):
    rotation_angles = [
        p.rotation_angle
        for f, p in plate_solutions.items()
        if f in self.light_frames
    ]
    mean = np.mean(rotation_angles)
    self.rotation_angle = mean

    self.rotation_angles_not_all_close = not np.allclose(
        rotation_angles, mean, rtol=0.02, atol=0)

  def store_width_height(self, fits_headers):
    self.width_pixels = fits_headers[self.light_frames[0]]['NAXIS1']
    self.height_pixels = fits_headers[self.light_frames[0]]['NAXIS2']

  def store_exposure_time_s(self, fits_headers):
    self.exposure_time_s = sum(
        fits_headers[f]['EXPOSURE'] for f in self.light_frames)

  def __str__(self):
    return (f'Stack({self.target}, {self.id}, {self.status}, '
            f'RA {ra_to_hms(self.center[0])}, '
            f'Dec {dec_to_dms(self.center[1])}, '
            f'{self.width_pixels}x{self.height_pixels}, '
            f'{self.image_scale:.2f}"/px, '
            f'{self.filter}, {len(self.light_frames)} light frames)')
