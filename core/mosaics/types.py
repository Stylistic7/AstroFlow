from typing import NewType, TypeAlias
from dataclasses import dataclass
from beartype import beartype
from shapely import box  # type: ignore

from geometry.projection import Projection
from hashing import hash_dataclass
from targets.types import TargetHash
from attributes import AstrometricAttributes, ChronologyAttributes
from cache.content_hashed import ContentHashed
from cache.context_ref import ContextRef

Template: TypeAlias = tuple[int | None, int | None]
Strip: TypeAlias = tuple[TargetHash, ...]

MosaicHash = NewType("MosaicHash", str)


@beartype
@dataclass(order=True, frozen=True)
class Mosaic(
    ContentHashed, AstrometricAttributes, ChronologyAttributes[MosaicHash]
):
    target_ids: tuple[TargetHash, ...]
    strips: tuple[Strip, ...]

    def __str__(self) -> str:
        return (
            f"Mosaic(name={self.name}, stable_hash={self.stable_hash}, "
            f"content_hash={self.content_hash}, {len(self.target_ids)} targets)"
        )

    def __hash__(self) -> int:
        assert self.content_hash, "Mosaic has empty content_hash"
        return hash_dataclass(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Mosaic):
            return NotImplemented

        return (
            self.stable_hash == other.stable_hash
            and self.content_hash == other.content_hash
        )

    # TODO: Add a valid() method, and assert on it.


class MosaicsDict(dict[MosaicHash, ContextRef["Mosaic"]]):
    def __hash__(self):
        from hashing import hash_list_int
        from cache.context_ref import dict_content_hash
        return hash_list_int([dict_content_hash(self)])

    def __str__(self):
        return f"MosaicsDict({len(self)})"


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
        return (
            f"Viewbox(content_hash={self.content_hash} "
            f"{self.num_pixels_x} px x {self.num_pixels_y} px"
        )

    def __hash__(self) -> int:
        return hash_dataclass(self)


@beartype
@dataclass(frozen=True)
class Mask(ContentHashed):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def to_box(self):
        return box(self.xmin, self.ymin, self.xmax, self.ymax)

    def __str__(self):
        return (
            f"Mask(content_hash={self.content_hash} {self.xmin=} "
            f"{self.xmax=} {self.ymin=} {self.ymax=}"
        )

    def __hash__(self) -> int:
        return hash_dataclass(self)
