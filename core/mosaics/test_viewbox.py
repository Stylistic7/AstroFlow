"""Unit tests for viewbox computation."""
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon
from frozendict import frozendict

from mosaics.viewbox import compute_viewbox
from previews.types import QualityLevel
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
    vb = compute_viewbox(QualityLevel.SUBFRAME, obj)
    # Subframe mode uses 2048px width regardless of image scale.
    assert vb.num_pixels_x == 2048


def test_viewbox_master_uses_image_scale():
    obj = _make_object(image_scale=2.21)
    vb = compute_viewbox(QualityLevel.MASTER, obj)
    # Master mode: pixels should be proportional to FOV / image_scale.
    # 2 degree FOV at 2.21 arcsec/pixel = ~3260 pixels.
    assert vb.num_pixels_x > 3000
    assert vb.num_pixels_x < 4000


def test_viewbox_dimensions_positive():
    obj = _make_object()
    for mode in [QualityLevel.SUBFRAME, QualityLevel.MASTER]:
        vb = compute_viewbox(mode, obj)
        assert vb.num_pixels_x > 0
        assert vb.num_pixels_y > 0
        assert vb.resolution > 0
        assert vb.x_span > 0
        assert vb.y_span > 0


def test_viewbox_tile_count():
    obj = _make_object()
    vb = compute_viewbox(QualityLevel.MASTER, obj)
    assert vb.num_tiles_x >= 1
    assert vb.num_tiles_y >= 1
    assert vb.tile_width == 2048
    assert vb.tile_height == 2048


def test_viewbox_wider_fov_more_pixels():
    narrow = _make_object(fov_deg=1.0)
    wide = _make_object(fov_deg=4.0)
    vb_narrow = compute_viewbox(QualityLevel.MASTER, narrow)
    vb_wide = compute_viewbox(QualityLevel.MASTER, wide)
    assert vb_wide.num_pixels_x > vb_narrow.num_pixels_x


def test_viewbox_downscaled_master_fewer_pixels():
    obj = _make_object(image_scale=2.21)
    vb_master = compute_viewbox(QualityLevel.MASTER, obj)
    vb_downscaled = compute_viewbox(QualityLevel.DOWNSCALED_MASTER, obj)
    # 8x downscale should produce ~1/8th the pixels in each dimension.
    ratio = vb_master.num_pixels_x / vb_downscaled.num_pixels_x
    assert 7 < ratio < 9
