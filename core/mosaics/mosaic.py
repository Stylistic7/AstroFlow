from dataclasses import dataclass, fields
from typing import Optional
import datetime
import numpy as np

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from hashing import make_hashable, hash_dataclass
from targets.types import TargetHash
from mosaics.types import TileIndices, MosaicHash


@dataclass(order=True, frozen=True)
class Mosaic:
  name: str

  content_hash: str
  stable_hash: str

  target_ids: list[TargetHash]
  tile_indices: list[TileIndices]
  strips: list[list[TargetHash]]

  oldest_date: Optional[datetime.datetime]
  newest_date: Optional[datetime.datetime]

  center: list[float]
  footprint: SingleSphericalPolygon
  projected_footprints: dict[str, Polygon]

  def __str__(self) -> str:
    return (
        f'Mosaic(name={self.name}, stable_hash={self.stable_hash}, '
        f'content_hash={self.content_hash}, {len(self.target_ids)} targets)')

  def __hash__(self) -> int:
    return hash_dataclass(self)

  def __eq__(self, other) -> bool:
    return (self.stable_hash == other.stable_hash and
            self.content_hash == other.content_hash)

  # TODO: Add a valid() method, and assert on it.


MosaicsDict = dict[MosaicHash, Mosaic]
