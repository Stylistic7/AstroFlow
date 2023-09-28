import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon


@dataclass
class Mosaic:
  id: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  target_ids: list[str] = field(default_factory=list)

  center: list[float] = field(default_factory=list)
  footprint: list[tuple[float]] = None
  viewbox: list[tuple[float]] = None

  def store_metadata(self, targets):
    # print('### Entering Mosaic.store_metadata')
    # TODO: Some of these other kinds of metadata might apply to Targets?
    # self.store_id()
    # self.store_exposure_time_s(fits_headers)
    self.store_center(targets)
    self.store_footprint(targets)
    self.store_viewbox()
    # self.store_min_footprint(plate_solutions)
    # self.store_aspect_ratio()

  def store_center(self, targets):
    centers = [t.center for t in targets if t.name in self.target_ids]
    mean_center = np.mean(centers, axis=0)
    self.center = list(mean_center)

  def store_footprint(self, targets):
    footprints = [
        t.footprint_polygon() for t in targets if t.name in self.target_ids
    ]
    footprint = functools.reduce(Polygon.union, footprints, footprints[0])
    self.footprint = list(footprint.exterior.coords)

  def store_viewbox(self):
    min_ra = np.min([p[0] for p in self.footprint])
    max_ra = np.max([p[0] for p in self.footprint])
    min_dec = np.min([p[1] for p in self.footprint])
    max_dec = np.max([p[1] for p in self.footprint])

    self.viewbox = [min_ra, min_dec, max_ra - min_ra, max_dec - min_dec]

  def footprint_polygon(self):
    return Polygon(self.footprint)
