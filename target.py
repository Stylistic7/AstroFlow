import numpy as np
from dataclasses import dataclass, field
import functools
import hashlib
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon
import xxhash

import mollweide
import footprints


def hash_string(s):
  return hashlib.sha256(bytes(s, 'utf-8')).hexdigest()


@dataclass(order=True)
class Target:
  id: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  # TODO: Rename to id
  name: str = None

  stack_ids: list[str] = field(default_factory=list)

  center: list[float] = field(default_factory=list)
  footprint: SphericalPolygon = None
  projected_footprint: dict[str, Polygon] = None

  def __str__(self):
    return (f'Target(name={self.name}, {len(self.stack_ids)} stacks)')

  def __hash__(self):
    return xxhash.xxh64(self.id).intdigest()

  # TODO: How to guarantee these are always in the same order?
  def store_id(self):
    self.id = hash_string('_'.join(self.stack_ids))
    self.sort_index = self.name

  def store_metadata(self, stacks):
    # TODO: Some of these other kinds of metadata might apply to Targets?
    self.store_id()
    self.store_center(stacks)
    self.store_footprint(stacks)

  def store_center(self, stacks):
    centers = [s.center for s in stacks.values() if s.id in self.stack_ids]
    mean_center = np.mean(centers, axis=0)
    self.center = list(mean_center)

  def store_footprint(self, stacks):
    self.footprint = next(
        SphericalPolygon.multi_intersection([
            SphericalPolygon(stacks[stack_id].footprint)
            for stack_id in self.stack_ids
        ]).points).tolist()
    self.projected_footprint = footprints.all_projections(
        SphericalPolygon(self.footprint))

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)
