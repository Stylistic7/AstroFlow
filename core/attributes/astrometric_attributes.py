from dataclasses import dataclass
from beartype import beartype
import numpy as np
import astropy  # type: ignore
from astropy.coordinates import SkyCoord
import astropy.units as u
from cdshealpix import nested as cds  # or ring
from collections.abc import Sequence, Iterable
from astropy.wcs import WCS  # type: ignore
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from typing import Any

from clustering.angles import angular_mean_mod_pi
from context import context
from metadata.classes import MetadataDict, FrameHash
import geometry.footprints as footprints
from catalogs import position_string


def healpix_cells_from_poly(poly, nside: int = 10):
    depth = int(np.log2(nside))  # cdshealpix uses depth = log2(nside)
    ra_deg, dec_deg = poly.to_radec()
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    ipix, _, _ = cds.polygon_search(c.ra, c.dec, depth, flat=True)
    return list(map(int, np.unique(ipix)))


# TODO: Start using radians everywhere, and strongly-type the units.
def spherical_centroid_deg(
    points: Iterable[tuple[float, float]],
) -> tuple[float, float]:
    x = y = z = 0.0
    n = 0

    for ra_deg, dec_deg in points:
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        c = np.cos(dec)
        x += c * np.cos(ra)
        y += c * np.sin(ra)
        z += np.sin(dec)
        n += 1

    if n == 0:
        raise ValueError("empty input")

    r = np.sqrt(x * x + y * y + z * z)
    if r == 0.0:
        raise ValueError("centroid undefined")

    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    dec = np.rad2deg(np.arcsin(z / r))

    return float(ra), float(dec)


# TODO: Unify this WCS stuff and put it somewhere more general.
# It's in target_identification and in mosaics/projection.
@beartype
def _spherical_polygon_for_wcs(wcs: WCS) -> SingleSphericalPolygon:
    footprint = SingleSphericalPolygon.from_wcs(wcs)
    if not footprint.is_clockwise():
        footprint = SingleSphericalPolygon(footprint.points[::-1])
    assert not SphericalPolygon.self_intersect(footprint.points)
    return footprint


def _intersect_polygons(
    polygons: list[SphericalPolygon],
) -> SingleSphericalPolygon:
    footprint = SphericalPolygon.multi_intersection(polygons)
    points = list(footprint.points)
    if len(points) != 1:
        raise ValueError("Object has invalid polygon footprint")
    footprint = SingleSphericalPolygon(points[0])
    if not footprint.is_clockwise():
        footprint = SingleSphericalPolygon(footprint.points[::-1])
    assert not SphericalPolygon.self_intersect(footprint.points)

    return footprint


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
        cls, frames: Sequence[FrameHash], metadata: MetadataDict
    ) -> "AstrometricAttributes":
        metadata_list = [
            i for i in metadata.values() if i.content_hash in frames
        ]
        sorted_metadata = sorted(
            metadata_list, key=lambda i: i.header["DATE-OBS"]
        )
        oldest_image_metadata = sorted_metadata[0]

        center = list(
            spherical_centroid_deg(
                [m.plate_solution.center for m in metadata.values()]
            )
        )
        name = position_string.canonical_target_string(center)
        image_scale = np.mean(
            [m.plate_solution.image_scale for m in metadata.values()]
        )
        rotation_angle = angular_mean_mod_pi(
            [m.plate_solution.rotation_angle for m in metadata.values()]
        )
        footprint = SingleSphericalPolygon.from_wcs(
            astropy.io.fits.Header(oldest_image_metadata.plate_solution.header)
        )
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

    @classmethod
    def from_objects(
        cls, objects: Sequence["AstrometricAttributes"]
    ) -> "AstrometricAttributes":
        # TODO: Strongly-type physical units, maybe do auto-conversion.
        center = list(spherical_centroid_deg([o.center for o in objects]))
        image_scale = np.mean([o.image_scale for o in objects])
        rotation_angle = angular_mean_mod_pi(
            [o.rotation_angle for o in objects]
        )
        name = position_string.canonical_target_string(center)
        footprint = _intersect_polygons(
            [SphericalPolygon(o.footprint) for o in objects]
        )
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
