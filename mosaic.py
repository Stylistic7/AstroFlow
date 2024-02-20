import numpy as np
from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon

from hashing import hash_list
import footprints


@dataclass(order=True)
class Mosaic:
  id: str = None

  name: str = None

  target_ids: list[str] = field(default_factory=list)
  tile_indices: list[tuple] = field(default_factory=list)

  center: list[float] = field(default_factory=list)
  footprint: SphericalPolygon = None
  projected_footprint: dict[str, Polygon] = None

  def __post_init__(self):
    self.id = self.hash()

  # TODO: Replace hash with __hash__ everywhere?
  def hash(self) -> str:
    return hash_list(self.target_ids)

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)

  def __str__(self):
    return (f'Mosaic(name={self.name}, id={self.id}, '
            f'{len(self.target_ids)} targets)')
