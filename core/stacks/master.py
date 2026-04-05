"""Master — the concrete artifact produced by stacking a Stack.

A Master carries its own AstrometricAttributes derived from the master
image's plate solution, which may differ from the source Stack's
pre-stacking footprint (e.g. after WBPP autocrop).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from beartype import beartype

from hashing import hash_list_int
from cache.content_hashed import ContentHashed
from attributes import (
    AstrometricAttributes,
    ChronologyAttributes,
    ImagingAttributes,
    PlateSolutionAttributes,
)
from stacks.types import StackHash
from metadata.classes import ImageMetadata
from units import Degrees, Radians, ArcsecondsPerPixel

import structlog

log = structlog.get_logger()


@beartype
@dataclass(frozen=True)
class Master(
    ContentHashed,
    AstrometricAttributes,
    ChronologyAttributes[StackHash],
    ImagingAttributes,
    PlateSolutionAttributes,
):
    """The result of stacking a Stack — has its own post-stacking astrometry."""
    stack_id: StackHash
    filename: Path

    @classmethod
    def from_stack_and_metadata(
        cls, stack, image_metadata: ImageMetadata,
    ) -> "Master":
        from stacks.composite_stack import Stack
        assert isinstance(stack, Stack)
        from attributes.astrometric_attributes import (
            healpix_cells_from_poly,
            _spherical_polygon_for_wcs,
        )
        from geometry import footprints
        from frozendict import frozendict
        from astropy.wcs import WCS
        import astropy.io.fits

        ps = image_metadata.plate_solution
        if ps is None:
            raise ValueError(
                f"Cannot create Master for stack {stack.stable_hash}: "
                f"master image has no plate solution"
            )

        wcs = WCS(astropy.io.fits.Header(ps.header))
        footprint = _spherical_polygon_for_wcs(wcs)
        projected_footprints = frozendict(footprints.all_projections(footprint))
        healpix_cells = healpix_cells_from_poly(footprint)

        return cls(
            # ContentHashed
            content_hash=image_metadata.content_hash,
            # AstrometricAttributes (from master's plate solution)
            name=stack.name,
            center=(Degrees(ps.center[0]), Degrees(ps.center[1])),
            image_scale=ps.image_scale,
            rotation_angle=ps.rotation_angle,
            footprint=footprint,
            healpix_cells=healpix_cells,
            projected_footprints=projected_footprints,
            # ChronologyAttributes (inherited from Stack)
            stable_hash=stack.stable_hash,
            oldest_date=stack.oldest_date,
            newest_date=stack.newest_date,
            # ImagingAttributes (inherited from Stack)
            image_type=stack.image_type,
            camera=stack.camera,
            filter=stack.filter,
            mean_temperature=stack.mean_temperature,
            exposure=stack.exposure,
            integration=stack.integration,
            width_pixels=image_metadata.width_pixels,
            height_pixels=image_metadata.height_pixels,
            # Master-specific
            stack_id=stack.stable_hash,
            filename=image_metadata.filename,
            plate_solution=ps,
        )

    def __hash__(self) -> int:
        return hash_list_int([self.content_hash])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Master):
            return NotImplemented
        return self.content_hash == other.content_hash

    def __str__(self) -> str:
        return (
            f"Master(stack_id={self.stack_id}, "
            f"center=({self.center[0]:.2f}, {self.center[1]:.2f}), "
            f"scale={self.image_scale:.2f}aspp, "
            f"{self.width_pixels}x{self.height_pixels})"
        )


class MastersDict(dict[StackHash, "Master"]):
    def __hash__(self):
        return hash_list_int(
            [self[k].content_hash for k in sorted(self.keys())]
        )

    def __str__(self):
        return f"MastersDict({len(self)})"
