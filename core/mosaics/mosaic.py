from dataclasses import dataclass, field
from typing import Optional
import datetime
import numpy as np

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from hashing import hash_list
from targets.types import TargetHash
from mosaics.types import TileIndices, MosaicHash


@dataclass(order=True, frozen=True)
class Mosaic:
  name: str

  content_hash: str
  stable_hash: str

  target_ids: list[TargetHash]
  tile_indices: list[TileIndices]
  origin: np.ndarray
  basis_vectors: np.ndarray

  oldest_date: Optional[datetime.datetime]
  newest_date: Optional[datetime.datetime]

  center: list[float]
  footprint: SingleSphericalPolygon
  projected_footprints: dict[str, Polygon]

  def __str__(self) -> str:
    return (
        f'Mosaic(name={self.name}, stable_hash={self.stable_hash}, '
        f'content_hash={self.content_hash}, {len(self.target_ids)} targets)')

  # TODO: Add a valid() method, and assert on it.


MosaicsDict = dict[MosaicHash, Mosaic]
