from dataclasses import dataclass, field
from typing import Optional
import datetime

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from hashing import hash_list


@dataclass(order=True)
class Target:
  name: str | None = None

  content_hash: Optional[str] = None
  stable_hash: Optional[str] = None

  stack_ids: list[str] = field(default_factory=list)

  oldest_date: Optional[datetime.datetime] = None
  newest_date: Optional[datetime.datetime] = None

  center: list[float] = field(default_factory=list)
  footprint: SingleSphericalPolygon = None
  projected_footprint: dict[str, Polygon] | None = None

  def __str__(self):
    return (f'Target(name={self.name}, {len(self.stack_ids)} stacks)')

  # TODO: Add a valid() method, and assert on it.
