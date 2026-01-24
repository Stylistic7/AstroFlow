from dataclasses import dataclass
from beartype import beartype
import numpy as np
import astropy  # type: ignore
from astropy.coordinates import SkyCoord
import astropy.units as u
from cdshealpix import nested as cds  # or ring
from typing import Any
from collections.abc import Sequence

from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore

from context import context
from metadata.classes import MetadataDict, FrameHash
import geometry.footprints as footprints
from catalogs import position_string


def _normalize_radians(x: float) -> float:
    if x < 0.0:
        return float(x + np.pi)
    else:
        return float(x)


def healpix_cells_from_poly(poly, nside: int = 10):
    depth = int(np.log2(nside))  # cdshealpix uses depth = log2(nside)
    ra_deg, dec_deg = poly.to_radec()
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    ipix, _, _ = cds.polygon_search(c.ra, c.dec, depth, flat=True)
    return list(map(int, np.unique(ipix)))


@beartype
@dataclass(frozen=True)
class AstrometricAttributes:
    name: str
    center: list[float]
    image_scale: float
    rotation_angle: float
    footprint: SingleSphericalPolygon
    healpix_cells: list[int]
    projected_footprints: dict[str, Any]  # TODO: Strongly-type this.

    @classmethod
    def from_frames(
        cls, frames: Sequence[FrameHash], metadata: MetadataDict | None = None
    ) -> "AstrometricAttributes":
        assert frames

        metadata = metadata or context.metadata
        metadata_list = [i for i in metadata.values() if i.content_hash in frames]

        sorted_metadata = sorted(metadata_list, key=lambda i: i.header["DATE-OBS"])
        oldest_image_metadata = sorted_metadata[0]

        assert oldest_image_metadata.plate_solution
        center = list(oldest_image_metadata.plate_solution.center)
        name = position_string.canonical_target_string(center)

        image_scale = oldest_image_metadata.plate_solution.image_scale
        rotation_angle = _normalize_radians(
            oldest_image_metadata.plate_solution.rotation_angle
        )

        # TODO: Unify this WCS stuff and put it somewhere more general.
        # It's in target_identification and in mosaics/projection.
        footprint = SingleSphericalPolygon.from_wcs(
            astropy.io.fits.Header(oldest_image_metadata.plate_solution.header)
        )
        if not footprint.is_clockwise():
            footprint = SingleSphericalPolygon(footprint.points[::-1])
        assert not SphericalPolygon.self_intersect(footprint.points)
        projected_footprints = footprints.all_projections(footprint)

        healpix_cells = healpix_cells_from_poly(footprint)

        return cls(
            name,
            center,
            image_scale,
            rotation_angle,
            footprint,
            healpix_cells,
            projected_footprints,
        )
