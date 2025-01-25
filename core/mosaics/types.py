from typing import NewType, TypeAlias
from dataclasses import dataclass, fields
from typing import Optional
import datetime
import numpy as np

from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore

from artifacts import Artifact
from geometry.projection import Projection
from hashing import hash_dataclass
from targets.types import TargetHash

Template: TypeAlias = tuple[int | None, int | None]
TileIndices: TypeAlias = tuple[int, int]
Pair: TypeAlias = list[TileIndices]

MosaicHash = NewType('MosaicHash', str)


@dataclass(order=True, frozen=True)
class Mosaic:
  name: str

  content_hash: str
  stable_hash: MosaicHash

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


@dataclass(frozen=True)
class Viewbox:
  projection: Projection
  x_min: float
  x_span: float
  y_min: float
  y_span: float

  def __str__(self):
    return 'Viewbox'


@dataclass(frozen=True)
class Panel:
  tile_indices: tuple[int, int]
  image: Artifact


@dataclass(frozen=True)
class MergeOp:
  input_images: list[Artifact]
  output_image: Artifact


@dataclass
class MosaicDimensions:
  width: int
  height: int
  min_ra: float
  ra_span: float
  min_dec: float
  dec_span: float
  ra_resolution: float
  dec_resolution: float
