import ray
import numpy as np

from attributes import AstrometricAttributes
from geometry.projection import Projection
from mosaics.types import Viewbox
from geometry.projections import mercator
from geometry.projections.gnomonic import GnomonicProjection

# TODO: Rename or unify geometry_compute_viewbox
from geometry.viewbox import compute_viewbox as geometry_compute_viewbox
from geometry.polygons import project_polygon
from previews.types import DOWNSCALE_FACTORS, SUBFRAME

_TILE_SIZE = 2048  # pixels


def _projection_for(object: AstrometricAttributes) -> Projection:
    """Gnomonic for a single field, Mercator for a real mosaic.

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

    Multi-panel mosaics keep Mercator. TAN's distortion grows as tan(distance
    from the tangent point), which is negligible over one field but not over a
    mosaic, and a mosaic has no single position angle to be tangent to. Giving
    the user real control here — centre of projection, projection family — is
    deliberate future work, gated behind a preview: switching a projection
    re-keys `compute_viewbox` and so rebuilds every placement and colour product
    downstream of it, which is GPU-hours, not seconds.

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
        return mercator.MercatorProjection()

    # Plain floats, not np.float64: the Viewbox's cache key is built from this
    # projection's REPR (`hashing.hash_list`), so the numbers in it have to
    # render identically everywhere.
    ra, dec = object.center
    return GnomonicProjection(
        float(np.deg2rad(float(ra))),
        float(np.deg2rad(float(dec))),
        -float(object.rotation_angle),
    )


def compute_viewbox(
    quality: str, object: AstrometricAttributes,
) -> Viewbox:
    def round_to_nearest_multiple(number: float, multiple: int) -> int:
        return int(np.ceil(number / multiple) * multiple)

    projection = _projection_for(object)

    # Project from spherical coordinates on the sky into 2D coordinates, in
    # radians. Compute the min and span in each dimension.
    xs, ys = project_polygon(object.footprint, projection)
    x_min, y_min, x_span, y_span = geometry_compute_viewbox(list(zip(xs, ys)))

    # Compute the number of pixels along the x axis. Non-subframe modes use
    # the object's image scale (adjusted by the quality's downscale factor);
    # subframe mode makes a fixed-size thumbnail.
    if quality == SUBFRAME:
        num_pixels_x = 2048
    else:
        effective_scale = object.image_scale * DOWNSCALE_FACTORS[quality]
        num_pixels_x = x_span / np.deg2rad(effective_scale / 3600)
    finished_pixels_x = int(np.ceil(num_pixels_x))

    # Round the number of pixels up to a multiple of the tile size.
    num_pixels_x = round_to_nearest_multiple(num_pixels_x, multiple=_TILE_SIZE)

    # Compute the effective resolution, with the rounded-up number of pixels.
    resolution = x_span / num_pixels_x

    # Compute the number of pixels in the y-axis, using the same resolution.
    num_pixels_y = y_span / resolution
    finished_pixels_y = int(np.ceil(num_pixels_y))
    num_pixels_y = round_to_nearest_multiple(num_pixels_y, multiple=_TILE_SIZE)

    num_tiles_y = int(np.ceil(num_pixels_y / _TILE_SIZE))
    num_tiles_x = int(np.ceil(num_pixels_x / _TILE_SIZE))

    return Viewbox(
        projection=projection,
        x_min=x_min,
        x_span=x_span,
        y_min=y_min,
        y_span=y_span,
        resolution=resolution,
        num_pixels_y=finished_pixels_y,
        num_pixels_x=finished_pixels_x,
        num_tiles_y=num_tiles_y,
        num_tiles_x=num_tiles_x,
        tile_height=_TILE_SIZE,
        tile_width=_TILE_SIZE,
    )
