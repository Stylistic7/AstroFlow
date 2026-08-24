import random
import warnings
from dataclasses import dataclass
from beartype import beartype
import numpy as np
import structlog
import astropy  # type: ignore
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS, FITSFixedWarning  # type: ignore
from cdshealpix import nested as cds  # or ring
from collections.abc import Sequence, Iterable

warnings.filterwarnings("ignore", category=FITSFixedWarning)
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
from typing import Any
from frozendict import frozendict

from clustering.angles import angular_mean_mod_pi
from context import context
from metadata.classes import MetadataDict, FrameHash
import geometry.footprints as footprints
from catalogs import position_string
from units import Degrees, Radians, ArcsecondsPerPixel

log = structlog.get_logger()


@beartype
def healpix_cells_from_poly(
    poly: SphericalPolygon | SingleSphericalPolygon, nside: int = 10
) -> tuple[int, ...]:
    depth = int(np.log2(nside))  # cdshealpix uses depth = log2(nside)
    ra_deg, dec_deg = poly.to_radec()
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    ipix, _, _ = cds.polygon_search(c.ra, c.dec, depth, flat=True)
    return tuple(map(int, np.unique(ipix)))


# TODO: Start using radians everywhere, and strongly-type the units.
@beartype
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


@beartype
def intersect_polygons(
    polygons: list[SphericalPolygon],
) -> SingleSphericalPolygon:
    """The region every input shares.

    This is the right footprint for a STACK. WBPP registers the subframes and
    crops the integration to where they all overlap, so the intersection is
    not a conservative estimate of the master's extent — it IS the master's
    extent, and claiming more would claim pixels the file does not have.
    """
    for i, p in enumerate(polygons):
        log.info(
            "Intersecting polygon",
            index=i,
            area=p.area(),
            num_sub_polygons=len(list(p.points)),
        )
    footprint = SphericalPolygon.multi_intersection(polygons)
    points = list(footprint.points)
    if len(points) != 1:
        raise ValueError(
            f"Polygon intersection produced {len(points)} sub-polygons "
            f"(expected 1) from {len(polygons)} input polygons. "
            f"Input areas: {[p.area() for p in polygons]}"
        )
    footprint = SingleSphericalPolygon(points[0])
    if not footprint.is_clockwise():
        footprint = SingleSphericalPolygon(footprint.points[::-1])
    assert not SphericalPolygon.self_intersect(footprint.points)

    return footprint


@beartype
def covering_polygon(
    polygons: list[SphericalPolygon],
) -> SingleSphericalPolygon:
    """One polygon containing every input — the CONVEX HULL of their
    vertices, not an exact union.

    This is the right footprint for a TARGET. Its members are separate
    per-filter masters that get projected onto one shared canvas, and nothing
    has cropped them to a common region, so the canvas has to hold all of
    them. (Contrast `intersect_polygons`, which is right for a Stack because
    WBPP already cropped its master to the overlap.)

    Exactness buys nothing here and costs everything. The consumer,
    `compute_viewbox`, takes the axis-aligned BOUNDING BOX of the projected
    footprint, and the bbox of a union equals the bbox of its convex hull —
    byte-identical canvases either way. Meanwhile
    `SphericalPolygon.multi_union` accumulates vertices pairwise and blew
    past a 60s timeout on 20 inputs. The hull does 20 in 6 ms, and always
    returns exactly one sub-polygon, so the degenerate multi-sub-polygon
    case a union has to raise on cannot arise at all.

    The cost is over-coverage where members are strongly rotated relative to
    one another: 1.17x the largest member's area across a 5 degree spread.
    Real spreads are far smaller — the Heart's six channels span 1.31 degrees
    — and over-coverage only ever means the canvas carries a little extra
    empty sky, which stays exactly 0.0 and reads as "no data" downstream.
    """
    # `[:-1]` drops each ring's closing duplicate before hulling, matching
    # `mosaics/identification.py`. The inner loop over `p.points` is
    # load-bearing separately: `SphericalPolygon.points` yields one array PER
    # SUB-POLYGON, so vstacking the generators themselves silently produces a
    # 1-wide array that `convex_hull` then indexes off.
    points = np.vstack([pts[:-1] for p in polygons for pts in p.points])

    footprint = _hull(points)

    # Shuffle-and-retry on self-intersection, lifted from
    # `mosaics/identification.py` where it has been reliable in production.
    # The seed comes from the points so the retry is reproducible — a random
    # seed would make the footprint, and every cache key downstream of it,
    # flap between processes.
    rng = random.Random(hash(tuple(map(tuple, points))))
    for i in range(10):
        if SphericalPolygon.self_intersect(footprint.points):
            shuffled = footprint.points
            rng.shuffle(shuffled)
            footprint = _hull(shuffled)
        else:
            break

    return footprint


def _hull(points: np.ndarray) -> SingleSphericalPolygon:
    """Convex hull, normalised to the un-complemented sense.

    `convex_hull`'s result depends on the order it receives points in: for one
    order it returns the field, for the reverse it returns the field's
    COMPLEMENT — the whole sky minus the field, area 4*pi. Two targets with
    identical geometry can land on opposite sides of this, and one did: it
    came back at 12.5657 sr instead of 0.0018, so mosaic identification
    rejected it for not being "roughly the same area" as the panel beside it,
    and found 0 mosaics from 2 overlapping frames.

    Re-hulling from the reversed INPUT is the fix. Reversing the OUTPUT points
    is not, and looks like it works: the complemented result is also
    degenerate — it comes back a vertex short — so flipping its points yields
    a clockwise triangle with exactly half the true area, which passes an
    `is_clockwise()` check while still being wrong.
    """
    hull = SingleSphericalPolygon.convex_hull(points)
    if hull.is_clockwise():
        return hull
    return SingleSphericalPolygon.convex_hull(points[::-1])


def _to_spherical_polygon(ssp: SingleSphericalPolygon) -> SphericalPolygon:
    """Build a `SphericalPolygon` from a `SingleSphericalPolygon`,
    bypassing the iteration path in `SphericalPolygon.__init__` that
    accesses `polygon._degenerate` directly. spherical_geometry 1.4.0
    introduced "return None for degenerate" semantics but didn't add a
    `getattr(..., '_degenerate', False)` fallback in its constructor,
    so any SSP that's missing that instance attribute (e.g. one
    unpickled from an older version) crashes the wrap. Passing the
    points array makes the constructor break out of the iteration and
    re-derive a single polygon from the points, which avoids the
    fragile attribute lookup entirely.
    """
    inside = getattr(ssp, "_inside", None)
    return SphericalPolygon(ssp.points, inside)


@beartype
@dataclass(frozen=True)
class AstrometricAttributes:
    name: str
    center: tuple[Degrees, Degrees]
    image_scale: ArcsecondsPerPixel
    rotation_angle: Radians
    footprint: SingleSphericalPolygon
    healpix_cells: tuple[int, ...]
    projected_footprints: frozendict[str, Any]

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

        for m in sorted_metadata:
            assert m.plate_solution is not None, (
                f"AstrometricAttributes.from_frames requires plate solutions; "
                f"{m.filename} has none (plate-solve failed?)"
            )

        center_raw = spherical_centroid_deg(
            [m.plate_solution.center for m in sorted_metadata]
        )
        center = (Degrees(center_raw[0]), Degrees(center_raw[1]))
        name = position_string.canonical_target_string(center)
        image_scale = ArcsecondsPerPixel(np.mean(
            [m.plate_solution.image_scale for m in sorted_metadata]
        ))
        rotation_angle = Radians(angular_mean_mod_pi(
            [m.plate_solution.rotation_angle for m in sorted_metadata]
        ))
        footprint = SingleSphericalPolygon.from_wcs(
            astropy.io.fits.Header(oldest_image_metadata.plate_solution.header)
        )
        projected_footprints = frozendict(footprints.all_projections(footprint))
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
        cls, objects: Sequence["AstrometricAttributes"],
        primary: "AstrometricAttributes",
        combine,
    ) -> "AstrometricAttributes":
        """Geometry for a composite, framed on ONE of its members.

        `primary` supplies the rotation and the image scale instead of a mean
        over the members. A circular mean is arithmetically right but frames
        the canvas on an angle no member actually has — the Heart's six
        channels span 1.31 degrees, so the mean matched none of them and every
        channel landed tilted. Framing on the reference channel makes THAT
        channel axis-aligned and leaves the others off by their own real
        offset, which is both honest and directly diagnosable in pixels.

        `combine` decides the footprint, and the right answer differs by
        composite. A Stack takes `intersect_polygons`, because WBPP already
        cropped its master to where the subframes overlap. A Target takes
        `covering_polygon`, because its per-filter masters are placed onto a
        shared canvas uncropped, and sizing that canvas to the intersection
        clips every channel extending past it — which it did, by 25-90 px per
        channel on the Heart. Pixels covered by some members but not others stay exactly 0.0
        (`place_image` fills with zero), so ragged edges are a DISPLAY
        concern, resolved by a coverage test at render time — not a reason to
        throw the data away here.

        Union is only safe because membership is already gated: identification
        groups members under an angular tolerance on both position and
        rotation, so a mis-solved frame cannot join and inflate the canvas.
        """
        center_raw = spherical_centroid_deg([o.center for o in objects])
        center = (Degrees(center_raw[0]), Degrees(center_raw[1]))
        image_scale = ArcsecondsPerPixel(primary.image_scale)
        rotation_angle = Radians(primary.rotation_angle)
        name = position_string.canonical_target_string(center)
        footprint = combine(
            [_to_spherical_polygon(o.footprint) for o in objects]
        )
        projected_footprints = frozendict(footprints.all_projections(footprint))
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projected_footprints",
            frozendict(self.projected_footprints),
        )
