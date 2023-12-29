import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon
import hashlib

import mollweide
import mercator


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
  footprints: dict[str, list[tuple[float]]] = None
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
    self.store_footprints(targets)
    self.store_footprint(targets)
    self.store_footprint_galactic()
    self.store_viewbox()
    # self.store_min_footprint(plate_solutions)
    # self.store_aspect_ratio()

  def store_center(self, targets):
    centers = [targets[id].center for id in self.target_ids]
    mean_center = np.mean(centers, axis=0)
    self.center = list(mean_center)

  def store_footprints(self, targets):
    projections = {
        'mollweide': mollweide.MollweideProjection(lon_0=np.pi),
        'mercator': mercator.MercatorProjection(lon_0=np.pi)
    }

    self.footprints = {}
    for name, proj in projections.items():
      print(f'### Projecting footprint with {name} projection')
      footprint = Polygon()
      for target_id in self.target_ids:
        coords = list(targets[target_id].footprint_polygon().exterior.coords)
        print('type(coords)', type(coords))
        print('type(coords[0])', type(coords[0]))
        print('coords', coords)
        lon, lat = zip(*coords)
        lon, lat = np.radians(np.array(lon)), np.radians(np.array(lat))
        print('type(lon)', type(lon))
        print('lon', lon)
        print('lat', lat)
        xx, yy = proj.project(lon, lat)
        print('xx', xx)
        print('yy', yy)
        target_footprint = Polygon(zip(xx, yy))
        footprint = Polygon.union(footprint, target_footprint)

      self.footprints[name] = footprint.exterior.coords

  def store_footprint(self, targets):
    footprints = [targets[id].footprint_polygon() for id in self.target_ids]
    footprint = functools.reduce(Polygon.union, footprints, footprints[0])
    self.footprint = list(footprint.exterior.coords)

  def store_footprint_galactic(self):
    proj = mollweide.MollweideProjection()
    self.footprint_galactic = []
    for ra, dec in self.footprint:
      lon, lat = mollweide.equatorial_to_galactic(ra, dec)
      x, y = proj.project(lon, lat)
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
