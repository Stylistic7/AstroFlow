from dataclasses import dataclass, field
from typing import Optional

from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon

from hashing import hash_list


@dataclass(order=True)
class Mosaic:
  name: str = None

  content_hash: Optional[str] = None
  geometric_hash: Optional[str] = None

  target_ids: list[str] = field(default_factory=list)
  tile_indices: list[tuple] = field(default_factory=list)

  center: list[float] = field(default_factory=list)
  footprint: SphericalPolygon = None
  projected_footprint: dict[str, Polygon] = None

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)

  def __str__(self):
    return (
        f'Mosaic(name={self.name}, geometric_hash={self.geometric_hash}, '
        f'content_hash={self.content_hash}, {len(self.target_ids)} targets)')

  # TODO: Add a valid() method, and assert on it.
