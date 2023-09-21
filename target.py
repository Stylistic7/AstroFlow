import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon


@dataclass
class Target:
  id: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  # TODO: Rename to id
  name: str = None

  stack_ids: list[str] = field(default_factory=list)

  footprint: list[tuple[float]] = None

  def __eq__(self, other):
    return self.light_frames == other.light_frames

  def __repr__(self):
    return (f'Target(name={self.name}, {len(self.stack_ids)} stacks)')

  def store_metadata(self, stacks):
    # TODO: Some of these other kinds of metadata might apply to Targets?
    # self.store_id()
    # self.store_exposure_time_s(fits_headers)
    self.store_center(stacks)
    self.store_footprint(stacks)
    # self.store_min_footprint(plate_solutions)
    # self.store_aspect_ratio()

  def store_center(self, stacks):
    centers = [s.center for s in stacks if s.id in self.stack_ids]
    mean_center = np.mean(centers, axis=0)
    self.center = list(mean_center)

  def store_footprint(self, stacks):
    footprints = [
        s.footprint_polygon() for s in stacks if s.id in self.stack_ids
    ]
    footprint = functools.reduce(Polygon.intersection, footprints,
                                 footprints[0])
    self.footprint = list(footprint.exterior.coords)

  def footprint_polygon(self):
    return Polygon(self.footprint)
