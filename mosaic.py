import numpy as np
from dataclasses import dataclass, field
import functools
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon
import hashlib

import projection
import mollweide
import mercator
import sinusoidal
import footprints


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
  footprint: SphericalPolygon = None
  projected_footprint: dict[str, Polygon] = None

  # TODO: How to guarantee these are always in the same order?
  def store_id(self):
    self.id = hash_string('_'.join(self.target_ids))
    self.sort_index = self.name

  def store_metadata(self, targets):
    self.store_id()
    self.store_center(targets)
    self.store_footprint(targets)

  def store_center(self, targets):
    centers = [targets[id].center for id in self.target_ids]
    mean_center = np.mean(centers, axis=0)
    self.center = list(mean_center)

  def store_footprint(self, targets):
    self.footprint = next(
        SphericalPolygon.multi_union([
            SphericalPolygon(targets[target_id].footprint)
            for target_id in self.target_ids
        ]).points).tolist()
    self.projected_footprint = footprints.all_projections(
        SphericalPolygon(self.footprint))

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)

  def __str__(self):
    return (f'Mosaic(name={self.name}, id={self.id}, '
            f'{len(self.target_ids)} targets)')
