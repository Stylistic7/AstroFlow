from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from hashing import hash_list


@dataclass(order=True)
class Target:
  id: str | None = None
  name: str | None = None

  stack_ids: list[str] = field(default_factory=list)

  center: list[float] = field(default_factory=list)
  footprint: SphericalPolygon = None
  projected_footprint: dict[str, Polygon] | None = None

  def __post_init__(self):
    self.id = self.hash()

  # TODO: Replace hash with __hash__ everywhere?
  def hash(self) -> str:
    return hash_list(self.stack_ids)

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)

  def __str__(self):
    return (f'Target(name={self.name}, {len(self.stack_ids)} stacks)')
