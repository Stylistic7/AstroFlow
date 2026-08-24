"""Unit tests for `compute_viewbox_measured` — the viewbox framed on the
channels' OWN plate solutions rather than the object's declared attributes.

The load-bearing assertions are the two failures that motivated the node:
the reference channel must come out axis-aligned (the declared/measured gap
tilted H by 3.38° on the Heart), and the canvas must contain every channel
(the declared intersection clipped 25-90 px per channel).

Ray-free: the node's raw body runs via `._function`, fed synthetic
`ImageMetadata` whose plate solutions are built from the same WCS math the
real solver produces.
"""
from pathlib import Path

import numpy as np
import pytest

from metadata.classes import ImageMetadata
from mosaics.viewbox_measured import compute_viewbox_measured
from plate_solving import (
    PlateSolution, get_image_scale, get_rotation_angle,
)
from units import Degrees


def _channel(
    filter: str, rotation_deg: float,
    ra: float = 38.2, dec: float = 61.45,
    scale_arcsec: float = 2.21, nx: int = 6048, ny: int = 3907,
) -> ImageMetadata:
    """An ImageMetadata carrying a synthetic MEASURED plate solution."""
    scale_deg = scale_arcsec / 3600.0
    r = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(r), np.sin(r)
    header = {
        "NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": ra, "CRVAL2": dec,
        "CRPIX1": nx / 2, "CRPIX2": ny / 2,
        # get_rotation_angle is atan2(CD1_2, CD1_1); positive-determinant CD,
        # the parity every solution in the library has (see _projection_for).
        "CD1_1": scale_deg * cos_r, "CD1_2": scale_deg * sin_r,
        "CD2_1": -scale_deg * sin_r, "CD2_2": scale_deg * cos_r,
    }
    ps = PlateSolution(
        header=header, num_stars=1000,
        center=(Degrees(ra), Degrees(dec)),
        image_scale=get_image_scale(header),
        rotation_angle=get_rotation_angle(header),
    )
    return ImageMetadata(
        content_hash=f"hash_{filter}_{rotation_deg}",
        stable_hash="deadbeef00000000",
        filename=Path(f"/nonexistent/{filter}.xisf"),
        header={"FILTER": filter},
        width_pixels=nx, height_pixels=ny, channels=1,
        dtype=np.dtype(np.float32),
        plate_solution=ps,
    )


# The Heart's real measured spread: H at 92.27 while others sit lower.
_HEART = [
    _channel("H", 92.2665),
    _channel("S", 90.9646),
    _channel("O", 90.8453),
    _channel("R", 90.1131),
]


def _needed_px(vb, channel):
    """How many canvas pixels this channel's field spans in x and y."""
    from geometry.polygons import project_polygon
    import astropy.io.fits
    from spherical_geometry.polygon import SingleSphericalPolygon
    fp = SingleSphericalPolygon.from_wcs(
        astropy.io.fits.Header(channel.plate_solution.header))
    xs, ys = project_polygon(fp, vb.projection)
    return ((max(xs) - min(xs)) / vb.resolution,
            (max(ys) - min(ys)) / vb.resolution)


def test_reference_channel_is_axis_aligned():
    """The projection is rotated to H's MEASURED angle — the master's own
    solve — so H lands square. Framing on the declared (pre-WBPP) attribute
    left H tilted by the declared/measured gap: 3.38 degrees on the Heart."""
    vb = compute_viewbox_measured._function("Master", True, *_HEART)
    assert vb.projection.rotation == pytest.approx(
        -float(_HEART[0].plate_solution.rotation_angle))
    # And square means the bounding box is tight: H's field needs exactly the
    # pixels of its own frame (up to the hull's rounding), not a rotated
    # frame's inflated box.
    need_x, need_y = _needed_px(vb, _HEART[0])
    assert need_x == pytest.approx(6048, rel=0.01)
    assert need_y == pytest.approx(3907, rel=0.01)


def test_canvas_contains_every_channel():
    """The cover guarantees no channel is clipped — the invariant the
    declared intersection violated by 25-90 px per channel."""
    vb = compute_viewbox_measured._function("Master", True, *_HEART)
    for c in _HEART:
        need_x, need_y = _needed_px(vb, c)
        assert need_x <= vb.num_pixels_x + 1e-6, c.header["FILTER"]
        assert need_y <= vb.num_pixels_y + 1e-6, c.header["FILTER"]


def test_resolution_is_the_reference_channels_measured_scale():
    """No DOWNSCALE_FACTORS multiplier: a measured scale already IS the
    file's scale, downscaled or not."""
    vb = compute_viewbox_measured._function("Master", True, *_HEART)
    actual = np.rad2deg(vb.resolution) * 3600
    assert actual == pytest.approx(
        float(_HEART[0].plate_solution.image_scale), rel=1e-3)


def test_mercator_for_multi_panel():
    from geometry.projections.mercator import MercatorProjection
    vb = compute_viewbox_measured._function("Master", False, *_HEART)
    assert isinstance(vb.projection, MercatorProjection)


def test_subframe_canvas_is_fixed_width():
    vb = compute_viewbox_measured._function("Subframe", True, *_HEART)
    assert vb.num_pixels_x == 2048


def test_missing_plate_solution_fails_loudly():
    """A channel that can't be placed must not be silently dropped from the
    canvas it would later be clipped by."""
    bad = _channel("O", 90.0)
    object.__setattr__(bad, "plate_solution", None)
    with pytest.raises(AssertionError, match="plate solutions"):
        compute_viewbox_measured._function("Master", True, _HEART[0], bad)
