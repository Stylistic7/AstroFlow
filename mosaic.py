import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon
import hashlib

import mollweide


def hash_string(s):
  return hashlib.sha256(bytes(s, 'utf-8')).hexdigest()


@dataclass(order=True)
class Mosaic:
  id: str = None

  name: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  target_ids: list[str] = field(default_factory=list)
  tile_indices: list[tuple] = field(default_factory=list)

  center: list[float] = field(default_factory=list)
  footprint: list[tuple[float]] = None
  footprint_galactic: list[tuple[float]] = None
  viewbox: list[tuple[float]] = None

  # TODO: How to guarantee these are always in the same order?
  def store_id(self):
    self.id = hash_string('_'.join(self.target_ids))
    self.sort_index = self.name

  def store_metadata(self, targets):
    # print('### Entering Mosaic.store_metadata')
    # TODO: Some of these other kinds of metadata might apply to Targets?
    self.store_id()
    # self.store_exposure_time_s(fits_headers)
    self.store_center(targets)
    self.store_footprint(targets)
    self.store_footprint_galactic()
    self.store_viewbox()
    # self.store_min_footprint(plate_solutions)
    # self.store_aspect_ratio()

  def store_center(self, targets):
    centers = [targets[id].center for id in self.target_ids]
    mean_center = np.mean(centers, axis=0)
    self.center = list(mean_center)

  def store_footprint(self, targets):
    footprints = [targets[id].footprint_polygon() for id in self.target_ids]
    footprint = functools.reduce(Polygon.union, footprints, footprints[0])
    self.footprint = list(footprint.exterior.coords)

  def store_footprint_galactic(self):
    self.footprint_galactic = []
    for ra, dec in self.footprint:
      lon, lat = mollweide.equatorial_to_galactic(ra, dec)
      x, y = mollweide.project_Mollweide(lon, lat)
      self.footprint_galactic.append((x, y))

  def store_viewbox(self):
    min_ra = np.min([p[0] for p in self.footprint])
    max_ra = np.max([p[0] for p in self.footprint])
    min_dec = np.min([p[1] for p in self.footprint])
    max_dec = np.max([p[1] for p in self.footprint])

    self.viewbox = [min_ra, min_dec, max_ra - min_ra, max_dec - min_dec]

  def footprint_polygon(self):
    return Polygon(self.footprint)

  def __str__(self):
    return (f'Mosaic(name={self.name}, {len(self.target_ids)} targets)')
