from dataclasses import dataclass, field
from typing import Optional
import datetime

from spherical_geometry.polygon import SingleSphericalPolygon
from shapely.geometry import Polygon

from hashing import hash_list


@dataclass(order=True)
class Mosaic:
  name: str = None

  # TODO: Perhaps rename stable_hash to "stable_hash", because that's really
  # its intent.
  content_hash: Optional[str] = None
  stable_hash: Optional[str] = None

  target_ids: list[str] = field(default_factory=list)
  tile_indices: list[tuple] = field(default_factory=list)

  oldest_date: Optional[datetime.datetime] = None
  newest_date: Optional[datetime.datetime] = None

  center: list[float] = field(default_factory=list)
  footprint: SingleSphericalPolygon = None
  projected_footprint: dict[str, Polygon] = None

  def __str__(self):
    return (
        f'Mosaic(name={self.name}, stable_hash={self.stable_hash}, '
        f'content_hash={self.content_hash}, {len(self.target_ids)} targets)')

  # TODO: Add a valid() method, and assert on it.
