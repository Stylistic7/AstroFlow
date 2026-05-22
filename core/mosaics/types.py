from typing import NewType, TypeAlias
from dataclasses import dataclass
from beartype import beartype
from shapely import box  # type: ignore

from geometry.projection import Projection
from hashing import hash_dataclass, hash_list_int
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
    # Same targets as `target_ids`, ordered for the user. Index + 1 is
    # the panel number shown on polygon labels in the mosaic viewer and
    # in the "Panel N" prefix of the Telescopius / text export — single
    # source of truth so the two can never disagree. Sort key is RA
    # ascending, which matches rise order for typical mosaics where
    # panels share a Dec band.
    panel_target_ids: tuple[TargetHash, ...]

    def panel_numbers(self) -> dict[TargetHash, int]:
        return {tid: i + 1 for i, tid in enumerate(self.panel_target_ids)}

    # User-pipeline conveniences (Phase B of the pipelines-as-Python work).
    # All delegate to existing helpers; nothing here adds new behaviour,
    # it's purely friendlier names for use inside `@pipeline` bodies.

    @property
    def id(self) -> MosaicHash:
        """Display alias for `stable_hash` — matches the
        `target.id` / `mosaic.id` shape user pipelines use as dict keys."""
        return self.stable_hash

    @property
    def targets(self) -> tuple:
        """Resolve `target_ids` to live `Target` objects from `context.targets`.

        Defensively unwraps `ContextRef` values — matches the
        `_unwrap` pattern the existing builders use, since
        `TargetsDict` is permissive about ContextRef vs raw Target.
        """
        from cache.context_ref import ContextRef
        from context import local_context as context
        return tuple(
            (v.get() if isinstance(v, ContextRef) else v)
            for v in (context.targets[tid] for tid in self.target_ids)
        )

    @property
    def filters(self) -> list[str]:
        """All filters present across this mosaic's targets, deterministic order."""
        from targets.target_channel import filters_for_target
        seen: set[str] = set()
        out: list[str] = []
        for t in self.targets:
            for f in filters_for_target(t):
                if f not in seen:
                    seen.add(f)
                    out.append(f)
        return out

    def viewbox(self, mode):
        """Pixel-space viewbox at the given quality level (`mode`).

        `mode` is required because viewbox geometry depends on the
        downscale factor — there's no sensible single default. Pipelines
        receive it as an arg from the cart dispatch.
        """
        from mosaics.viewbox import compute_viewbox
        return compute_viewbox(mode, self)

    def __str__(self) -> str:
        return (
            f"Mosaic(name={self.name}, stable_hash={self.stable_hash}, "
            f"content_hash={self.content_hash}, {len(self.target_ids)} targets)"
        )

    def __hash__(self) -> int:
        assert self.content_hash, "Mosaic has empty content_hash"
        # content_hash is precomputed and uniquely identifies this
        # Mosaic. Hashing the whole dataclass — chasing strips,
        # targets, polygons, healpix_cells, footprint — is O(mosaic
        # size). When this gets called once per Node created in a
        # pipeline trace, a big mosaic blows up to 20+ seconds.
        return hash_list_int([self.content_hash])

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
