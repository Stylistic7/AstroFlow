from dataclasses import dataclass, field
from typing import Optional
import datetime

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from hashing import hash_list
from targets.types import TargetHash
from stacks.types import StackHash


@dataclass(order=True)
class Target:
  name: str

  content_hash: str
  stable_hash: TargetHash

  stack_ids: list[StackHash]

  oldest_date: datetime.datetime
  newest_date: datetime.datetime

  center: list[float]
  footprint: SingleSphericalPolygon
  projected_footprint: dict[str, Polygon]

  def __str__(self):
    return (f'Target(name={self.name}, {len(self.stack_ids)} stacks)')

  # TODO: Add a valid() method, and assert on it.
