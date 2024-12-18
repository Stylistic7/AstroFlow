from dataclasses import dataclass, fields
from typing import Optional
import datetime

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from hashing import make_hashable
from targets.types import TargetHash
from stacks.types import StackHash
from stacks.astrometric_attributes import AstrometricAttributes


@dataclass(order=True, frozen=True)
class Target(AstrometricAttributes):
  content_hash: str
  stable_hash: TargetHash

  stack_ids: list[StackHash]

  oldest_date: datetime.datetime
  newest_date: datetime.datetime

  def __str__(self) -> str:
    return (f'Target(target={self.target}, {len(self.stack_ids)} stacks)')

  def __hash__(self):
    return hash(
        tuple(make_hashable(getattr(self, f.name)) for f in fields(self)))

  # TODO: Add a valid() method, and assert on it.


TargetsDict = dict[TargetHash, Target]
