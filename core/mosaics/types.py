from typing import NewType, TypeAlias
from dataclasses import dataclass
from beartype import beartype

from geometry.projection import Projection
from hashing import hash_dataclass
from targets.types import TargetHash
from attributes import AstrometricAttributes, ChronologyAttributes
from task.content_hashed import ContentHashed

Template: TypeAlias = tuple[int | None, int | None]
TileIndices: TypeAlias = tuple[int, int]
Pair: TypeAlias = list[TileIndices]
Strip: TypeAlias = list[TargetHash]

MosaicHash = NewType('MosaicHash', str)


@beartype
@dataclass(order=True, frozen=True)
class Mosaic(ContentHashed, AstrometricAttributes,
             ChronologyAttributes[MosaicHash]):
  stable_hash: MosaicHash
  target_ids: list[TargetHash]
  strips: list[Strip]

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


@beartype
@dataclass(frozen=True)
class Viewbox(ContentHashed):
  projection: Projection

  x_min: float  # radians
  x_span: float  # radians

  y_min: float  # radians
  y_span: float  # radians

  resolution: float  # radians per pixel

  num_pixels_y: int
  num_pixels_x: int

  num_tiles_y: int
  num_tiles_x: int

  tile_height: int
  tile_width: int

  def __str__(self):
    return 'Viewbox'

  def __hash__(self) -> int:
    return hash_dataclass(self)
