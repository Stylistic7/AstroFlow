"""The viewbox as a graph node, computed from MEASURED channel geometry.

`compute_viewbox` (mosaics/viewbox.py) frames the canvas from the object's
DECLARED attributes — circular means over the raw subframes' plate solutions,
taken before WBPP ran. The masters being placed carry their own solutions,
and the two disagree: on the Heart, H's stack attribute says 88.887° while
the H master on disk solves at 92.2665°, so the "reference-framed" canvas
still tilted every channel by 3.38°.

This module frames the canvas from the channels' OWN plate solutions — the
measured WCS of the files `place_image` is about to project. Declared
geometry keeps deciding MEMBERSHIP (identification groups frames before any
master exists); measured geometry decides PIXELS.

It is a plain `@node` — no `identity=`, so no image lifecycle — returning a
`Viewbox`. Called inside a pipeline trace with channel HANDLES, it records
like any other node: the handles' `content_hash` keeps `filename_fn` path
prediction working at trace time, and the runtime resolves them to real
`ImageMetadata` before the body runs. The viewbox therefore re-keys exactly
when a channel master changes, which is the point.

Quality here is only "Subframe" (fixed 2048 px thumbnail canvas) or native.
The declared path multiplied the object's native scale by
`DOWNSCALE_FACTORS[quality]` because it couldn't see the files; a measured
channel's scale already IS the downscaled scale for downscaled pipelines,
so applying a factor again would double it.
"""
import astropy.io.fits
import numpy as np
from beartype import beartype
from spherical_geometry.polygon import SingleSphericalPolygon

from attributes.astrometric_attributes import (
    covering_polygon, spherical_centroid_deg, _to_spherical_polygon,
)
from decorators import node
from geometry.projections import mercator
from geometry.projections.gnomonic import GnomonicProjection
from geometry.polygons import project_polygon
from geometry.viewbox import compute_viewbox as geometry_compute_viewbox
from metadata.classes import ImageMetadata
from mosaics.types import Viewbox
from previews.types import SUBFRAME

# Which channel frames the canvas — the one carrying the structure, so it is
# the one that comes out axis-aligned. Same fixed rule as
# `targets/types._FILTER_PRIORITY` (H narrowband, L for LRGB, G for RGB),
# repeated here because that one ranks Stack objects and this ranks the
# FILTER strings on channel metadata; both must stay fixed rules — anything
# data-dependent would re-key the viewbox as data arrives.
_FILTER_PRIORITY = ("H", "L", "G", "S", "O", "R", "B")


def _rank(filter: str) -> tuple[int, str]:
    filter = str(filter or "")
    known = (_FILTER_PRIORITY.index(filter)
             if filter in _FILTER_PRIORITY else len(_FILTER_PRIORITY))
    return (known, filter)


def _measured_footprint(channel: ImageMetadata) -> SingleSphericalPolygon:
    """The channel master's sky footprint from its OWN plate solution — the
    extent of the pixels that exist in the file, WBPP crop included."""
    return SingleSphericalPolygon.from_wcs(
        astropy.io.fits.Header(channel.plate_solution.header)
    )


@beartype
@node
def compute_viewbox_measured(
    quality: str,
    single_panel: bool,
    *channels: ImageMetadata,
) -> Viewbox:
    """One shared canvas for `channels`, framed on the reference channel.

    * rotation + image scale: the reference channel's plate solution — the
      master's measured WCS, so the reference channel lands axis-aligned and
      unrotated, and the others show their own real offsets;
    * extent: the covering hull of every channel's measured footprint, so
      the canvas holds all the data (partial-coverage edges stay exact 0.0
      downstream — a display concern, not a reason to crop);
    * center: the spherical centroid of the channels' plate centers;
    * projection: TAN at the center, rotated to the reference (a Target —
      one pointing — reproduces what the camera saw); Mercator for a real
      multi-panel mosaic, which has no single position angle to be tangent
      to. Same reasoning, measured inputs — see `_projection_for`'s
      docstring in mosaics/viewbox.py for the sign convention and the
      sec(delta) cost argument.

    `channels` must all carry plate solutions; a channel without one cannot
    be placed anyway, and silently dropping it here would size a canvas that
    clips it later.
    """
    assert channels, "viewbox needs at least one channel"
    for c in channels:
        assert c.plate_solution is not None, (
            f"compute_viewbox_measured needs plate solutions; "
            f"{c.filename} has none — its producer must plate-solve."
        )

    reference = min(
        channels, key=lambda c: _rank(c.header.get("FILTER", "")))
    rotation = float(reference.plate_solution.rotation_angle)
    image_scale = float(reference.plate_solution.image_scale)

    center_raw = spherical_centroid_deg(
        [c.plate_solution.center for c in channels])

    footprint = covering_polygon(
        [_to_spherical_polygon(_measured_footprint(c)) for c in channels])

    if single_panel:
        # Plain floats: the Viewbox cache key is built from this projection's
        # REPR, so the numbers must render identically everywhere.
        projection = GnomonicProjection(
            float(np.deg2rad(center_raw[0])),
            float(np.deg2rad(center_raw[1])),
            -rotation,
        )
    else:
        projection = mercator.MercatorProjection()

    xs, ys = project_polygon(footprint, projection)
    x_min, y_min, x_span, y_span = geometry_compute_viewbox(list(zip(xs, ys)))

    if quality == SUBFRAME:
        num_pixels_x = 2048.0
    else:
        # The measured scale IS the file's scale — downscaled masters carry
        # their downscaled solution, so no DOWNSCALE_FACTORS multiplier
        # (applying one here would double-count the downscale).
        num_pixels_x = x_span / np.deg2rad(image_scale / 3600)
    finished_pixels_x = int(np.ceil(num_pixels_x))

    # Resolution follows the canvas actually returned — same invariant as
    # `compute_viewbox` after the tile-rounding fix: the canvas must contain
    # `x_span / resolution` pixels of field on both axes.
    resolution = x_span / finished_pixels_x
    finished_pixels_y = int(np.ceil(y_span / resolution))

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
