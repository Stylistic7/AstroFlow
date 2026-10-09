import numpy as np

from attributes import AstrometricAttributes
from geometry.projection import Projection
from mosaics.types import Viewbox
from geometry.projections.gnomonic import GnomonicProjection
from geometry.projections.stereographic import StereographicProjection

# TODO: Rename or unify geometry_compute_viewbox
from geometry.viewbox import compute_viewbox as geometry_compute_viewbox
from geometry.polygons import project_polygon
from previews.types import SUBFRAME



def _projection_for(object: AstrometricAttributes) -> Projection:
    """Gnomonic for a single field, stereographic for a real mosaic.

    A Target is one telescope pointing, so the projection that reproduces what
    the camera saw is TAN tangent at its own centre, rotated to its own position
    angle — the frame comes out axis-aligned, uniformly scaled, filling the
    canvas the way the raw subs filled the sensor. Under Mercator it lands
    north-up — so a field shot at PA 88.8 arrives rotated, a landscape sensor
    coming out portrait — and tapered by sec(delta): 8.6% across the Heart's
    2.5 degrees at delta +61.5. Wedge corners appear too, but only at
    intermediate position angles: at PA 30 the bounding box is 53% filled,
    while at PA 88.8 a rotated rectangle is very nearly a transposed one and
    the canvas is still ~95% data.

    The taper is the visible symptom; the expensive one is scale. Mercator's
    local scale IS sec(delta), and the sizing below divides the projected span
    by the field's TRUE angular resolution — so the canvas inflates by
    sec^2(delta) in area. Measured over the 442 plate-solved channel masters:
    Mercator asks for 1.0x native at the equator but 4.66-4.77x at the Heart's
    +61.5, i.e. the same image upsampled 2.1x per axis, carrying no more detail
    and costing 4.7x the pixels through every GPU stage after it. Gnomonic asks
    for 1.00x native at every declination in the library; summed over all 442
    masters the canvas drops from 44.4 to 24.3 gigapixels.

    Multi-panel mosaics use STEREOGRAPHIC, centred on the mosaic (2026-10-09).
    TAN's distortion grows as sec^2 of the distance from the tangent point
    (1.17 at 22.5 deg, the edge of the 45 deg span cap); STG's grows as
    sec^2(c/2) (1.04 there), and it is conformal, so stars stay round. Mosaics
    used to keep Mercator, centred at its default meridian RA 0h with no wrap:
    a mosaic straddling RA 0h (Sh2-171) got a 360-deg-wide, 586,150 px canvas,
    and its merges were memory-killed at 250-380 GB eleven times in one night.
    A centred projection has no wrap and no pole. Switching it re-keys
    `compute_viewbox` and rebuilds every placement and colour product
    downstream.

    The rotation is NEGATED, and that sign is measured rather than derived.
    Over all 1,589 plate solutions in the library — every one of which has a
    CD matrix with positive determinant, i.e. `s * [[cos, sin], [-sin, cos]]` —
    `-rotation_angle` frames the field with mean tightness (polygon area over
    bounding-box area) 0.9993, worst case 0.9919; `+rotation_angle` gives mean
    0.8967, worst case 0.4832. A synthetic CD matrix of the opposite parity
    reverses that conclusion, so the test builds the real one.

    Getting this sign wrong does NOT misplace pixels — `place_tile` unprojects
    each canvas pixel and looks it up through the source WCS, so any bijective
    projection lands the data correctly. What a wrong rotation costs is
    framing: the field arrives tilted and the bounding box inflates.
    """
    from targets.types import Target

    if not isinstance(object, Target):
        # A mosaic: centred, north-up unless it records a position angle.
        ra, dec = object.center
        return StereographicProjection(
            float(np.deg2rad(float(ra))),
            float(np.deg2rad(float(dec))),
            -float(getattr(object, "rotation_angle", 0.0) or 0.0),
        )

    # Plain floats, not np.float64: the Viewbox's cache key is built from this
    # projection's REPR (`hashing.hash_list`), so the numbers in it have to
    # render identically everywhere.
    ra, dec = object.center
    return GnomonicProjection(
        float(np.deg2rad(float(ra))),
        float(np.deg2rad(float(dec))),
        -float(object.rotation_angle),
    )


# A mosaic canvas this large is a geometry bug, not a mosaic: a 45-deg mosaic at
# 2.2"/px is ~5 Gpx. The merge holds three full-canvas float32 arrays, so a
# runaway canvas (the RA-wrap bug: 35 Gpx, 141 GB each) must fail here, loudly,
# instead of as a memory kill an hour into the merge.
MAX_CANVAS_PIXELS = 10_000_000_000


def check_canvas(nx: int, ny: int, what) -> None:
    assert nx * ny <= MAX_CANVAS_PIXELS, (
        f"canvas {nx} x {ny} px ({nx * ny / 1e9:.1f} Gpx) for {what} exceeds "
        f"{MAX_CANVAS_PIXELS / 1e9:.0f} Gpx: a projection or footprint bug, not a mosaic")


def compute_viewbox(
    quality: str, object: AstrometricAttributes,
) -> Viewbox:
    projection = _projection_for(object)

    # Project from spherical coordinates on the sky into 2D coordinates, in
    # radians. Compute the min and span in each dimension.
    xs, ys = project_polygon(object.footprint, projection)
    x_min, y_min, x_span, y_span = geometry_compute_viewbox(list(zip(xs, ys)))

    # Compute the number of pixels along the x axis. Non-subframe modes use
    # the object's image scale; subframe mode makes a fixed-size thumbnail.
    if quality == SUBFRAME:
        num_pixels_x = 2048
    else:
        num_pixels_x = x_span / np.deg2rad(object.image_scale / 3600)
    finished_pixels_x = int(np.ceil(num_pixels_x))

    # Resolution follows the canvas actually returned. This used to divide
    # x_span by a TILE-ROUNDED width while returning the un-rounded one, so
    # `resolution` was finer than the width it was paired with by exactly the
    # rounding factor — and the field then spanned more pixels than the canvas
    # had. The Heart's H merge came out 6174 px wide against a field needing
    # 8192 at its own resolution, clipping 24.6% off the right edge. Height
    # escaped because it is derived from `resolution`, which is why the bug
    # presented as "cropped on the right" only.
    #
    # Nothing needed the width to be a tile multiple. Chunking is a placement
    # concern and now lives there (`mosaics/placement/astroflow.py`), where
    # the last chunk simply overhangs. Rounding into `resolution` also
    # silently upsampled every canvas by up to 1.33x per axis — carrying no
    # more detail at 1.77x the pixels through every GPU stage — which is the
    # same inflation `_projection_for` rejects Mercator for.
    resolution = x_span / finished_pixels_x

    finished_pixels_y = int(np.ceil(y_span / resolution))
    check_canvas(finished_pixels_x, finished_pixels_y, object)

    return Viewbox(
        projection=projection,
        x_min=x_min,
        x_span=x_span,
        y_min=y_min,
        y_span=y_span,
        resolution=resolution,
        num_pixels_y=finished_pixels_y,
        num_pixels_x=finished_pixels_x,
    )
