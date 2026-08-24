"""Unit tests for viewbox computation."""
import numpy as np
import pytest
from spherical_geometry.polygon import SingleSphericalPolygon
from frozendict import frozendict

from mosaics.viewbox import compute_viewbox
from attributes.astrometric_attributes import AstrometricAttributes
from units import Degrees, ArcsecondsPerPixel, Radians


def _make_object(
    ra=100.0, dec=0.0, fov_deg=2.0, image_scale=2.21,
) -> AstrometricAttributes:
    """Create an AstrometricAttributes with a square footprint centered at (ra, dec)."""
    half = fov_deg / 2.0
    footprint = SingleSphericalPolygon.from_radec(
        [ra - half, ra + half, ra + half, ra - half],
        [dec - half, dec - half, dec + half, dec + half],
        center=[ra, dec],
    )
    return AstrometricAttributes(
        name="test",
        center=(Degrees(ra), Degrees(dec)),
        image_scale=ArcsecondsPerPixel(image_scale),
        rotation_angle=Radians(0.0),
        footprint=footprint,
        healpix_cells=tuple(),
        projected_footprints=frozendict(),
    )


def test_viewbox_subframe_has_fixed_width():
    obj = _make_object()
    vb = compute_viewbox("Subframe", obj)
    # Subframe mode uses 2048px width regardless of image scale.
    assert vb.num_pixels_x == 2048


def test_viewbox_master_uses_image_scale():
    obj = _make_object(image_scale=2.21)
    vb = compute_viewbox("Master", obj)
    # Master mode: pixels should be proportional to FOV / image_scale.
    # 2 degree FOV at 2.21 arcsec/pixel = ~3260 pixels.
    assert vb.num_pixels_x > 3000
    assert vb.num_pixels_x < 4000


def test_viewbox_dimensions_positive():
    obj = _make_object()
    for mode in ["Subframe", "Master"]:
        vb = compute_viewbox(mode, obj)
        assert vb.num_pixels_x > 0
        assert vb.num_pixels_y > 0
        assert vb.resolution > 0
        assert vb.x_span > 0
        assert vb.y_span > 0


def test_viewbox_canvas_contains_the_field():
    """The canvas must hold the whole projected footprint.

    Regression: `resolution` was derived from a tile-rounded width while the
    un-rounded width was returned, so the field spanned `x_span/resolution`
    pixels against a narrower canvas — 8192 vs 6174 on the Heart, clipping
    24.6% off the right edge of every master. Height was computed FROM
    `resolution` and so never clipped, which is why only x is a real risk;
    both are asserted so a future edit can't invert that.
    """
    for fov in (0.5, 1.0, 2.5, 5.0):
        for mode in ("Subframe", "Master"):
            vb = compute_viewbox(mode, _make_object(fov_deg=fov))
            assert vb.x_span / vb.resolution <= vb.num_pixels_x + 1e-6, (
                f"canvas {vb.num_pixels_x} px narrower than the "
                f"{vb.x_span / vb.resolution:.1f} px field at fov={fov} {mode}")
            assert vb.y_span / vb.resolution <= vb.num_pixels_y + 1e-6


def test_viewbox_resolution_is_native_scale():
    """No silent upsampling: a Master canvas samples at the object's own
    image scale, not finer. Rounding the width up to a 2048 multiple and
    dividing x_span by THAT inflated every canvas by up to 1.33x per axis."""
    obj = _make_object()
    vb = compute_viewbox("Master", obj)
    actual_arcsec = np.rad2deg(vb.resolution) * 3600
    assert actual_arcsec == pytest.approx(obj.image_scale, rel=1e-3)


def test_viewbox_wider_fov_more_pixels():
    narrow = _make_object(fov_deg=1.0)
    wide = _make_object(fov_deg=4.0)
    vb_narrow = compute_viewbox("Master", narrow)
    vb_wide = compute_viewbox("Master", wide)
    assert vb_wide.num_pixels_x > vb_narrow.num_pixels_x


def test_viewbox_downscaled_master_fewer_pixels():
    obj = _make_object(image_scale=2.21)
    vb_master = compute_viewbox("Master", obj)
    vb_downscaled = compute_viewbox("DownscaledMaster", obj)
    # 8x downscale should produce ~1/8th the pixels in each dimension.
    ratio = vb_master.num_pixels_x / vb_downscaled.num_pixels_x
    assert 7 < ratio < 9
