"""Test reliable polygon intersection with synthetic frames.

The spherical_geometry library's pairwise intersection is O(n²) in vertex
count — each step produces more vertices, slowing subsequent steps.
For 30 polygons spanning ±5° rotation it takes ~35s. This is acceptable
for intersecting Run footprints (typically 2-10 per Stack), but NOT for
intersecting hundreds of individual frames. The Stack footprint should
be computed from Run footprints, not individual frames.
"""
import numpy as np
import pytest
from astropy.wcs import WCS
from spherical_geometry.polygon import SingleSphericalPolygon, SphericalPolygon

from attributes.astrometric_attributes import _spherical_polygon_for_wcs, _intersect_polygons


def _polygon_for_rotation(rotation_deg: float, **kwargs) -> SingleSphericalPolygon:
    """Build a spherical polygon from a synthetic WCS at the given rotation."""
    ra = kwargs.get("ra", 300.0)
    dec = kwargs.get("dec", 40.0)
    scale_arcsec = kwargs.get("image_scale", 2.2)
    width = kwargs.get("width_pixels", 9576)
    height = kwargs.get("height_pixels", 6388)

    scale_deg = scale_arcsec / 3600.0
    rot_rad = np.radians(rotation_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)

    w = WCS(naxis=2)
    w.wcs.crpix = [width / 2, height / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.cd = [
        [scale_deg * cos_r, -scale_deg * sin_r],
        [scale_deg * sin_r, scale_deg * cos_r],
    ]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.pixel_shape = (width, height)
    return _spherical_polygon_for_wcs(w)


def test_intersect_two_rectangles():
    """Two slightly rotated rectangles should produce a valid intersection."""
    a = _polygon_for_rotation(0.0)
    b = _polygon_for_rotation(3.0)
    result = _intersect_polygons([SphericalPolygon(a), SphericalPolygon(b)])
    assert result.area() > 0
    assert result.area() < a.area()


def test_intersect_ten_rectangles():
    """Ten rectangles with rotations spanning ±5° should intersect cleanly."""
    angles = np.linspace(-5, 5, 10)
    polygons = [SphericalPolygon(_polygon_for_rotation(a)) for a in angles]
    result = _intersect_polygons(polygons)
    assert result.area() > 0
    individual_areas = [_polygon_for_rotation(a).area() for a in angles]
    assert result.area() < min(individual_areas)


@pytest.mark.timeout(60)
def test_intersect_20_rectangles():
    """20 rectangles — realistic upper bound for Run count per Stack."""
    angles = np.linspace(-5, 5, 20)
    polygons = [SphericalPolygon(_polygon_for_rotation(a)) for a in angles]
    result = _intersect_polygons(polygons)
    assert result.area() > 0


def test_intersection_area_converges():
    """Area should stabilize quickly — diminishing returns after ~5 polygons."""
    areas = []
    for n in [2, 5, 10]:
        angles = np.linspace(-5, 5, n)
        polygons = [SphericalPolygon(_polygon_for_rotation(a)) for a in angles]
        result = _intersect_polygons(polygons)
        areas.append(result.area())

    # After 5 polygons, area should be within 1% of the 10-polygon result
    assert abs(areas[1] - areas[2]) / areas[2] < 0.01


def test_intersection_smaller_than_all_inputs():
    """The intersection's area must be smaller than every input polygon."""
    angles = np.linspace(-3, 3, 5)
    polygons = [_polygon_for_rotation(a) for a in angles]
    result = _intersect_polygons([SphericalPolygon(p) for p in polygons])
    for poly in polygons:
        assert result.area() < poly.area()
