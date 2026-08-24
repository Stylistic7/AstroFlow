"""Test reliable polygon union with synthetic frames.

The composite footprint is the UNION of its members, so the canvas holds
every member's data rather than only where they all overlap.

The spherical_geometry library's pairwise combination is O(n²) in vertex
count — each step produces more vertices, slowing subsequent steps. This is
acceptable for Run footprints (typically 2-10 per Stack), but NOT for
hundreds of individual frames. The Stack footprint should be computed from
Run footprints, not individual frames.
"""
import numpy as np
import pytest
from astropy.wcs import WCS
from spherical_geometry.polygon import SingleSphericalPolygon, SphericalPolygon

from attributes.astrometric_attributes import (
    _spherical_polygon_for_wcs, covering_polygon, intersect_polygons,
)


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


def test_covering_two_rectangles():
    """Two slightly rotated rectangles union into something holding both."""
    a = _polygon_for_rotation(0.0)
    b = _polygon_for_rotation(3.0)
    result = covering_polygon([SphericalPolygon(a), SphericalPolygon(b)])
    assert result.area() > a.area()
    assert result.area() > b.area()


def test_covering_ten_rectangles():
    """Ten rectangles with rotations spanning ±5° should union cleanly."""
    angles = np.linspace(-5, 5, 10)
    polygons = [SphericalPolygon(_polygon_for_rotation(a)) for a in angles]
    result = covering_polygon(polygons)
    individual_areas = [_polygon_for_rotation(a).area() for a in angles]
    assert result.area() > max(individual_areas)


@pytest.mark.timeout(60)
def test_covering_20_rectangles():
    """20 rectangles — realistic upper bound for Run count per Stack."""
    angles = np.linspace(-5, 5, 20)
    polygons = [SphericalPolygon(_polygon_for_rotation(a)) for a in angles]
    result = covering_polygon(polygons)
    assert result.area() > 0


def test_covering_area_converges():
    """Area should stabilize quickly — the extremes of the rotation range
    dominate, so adding interior angles barely grows the hull."""
    areas = []
    for n in [2, 5, 10]:
        angles = np.linspace(-5, 5, n)
        polygons = [SphericalPolygon(_polygon_for_rotation(a)) for a in angles]
        result = covering_polygon(polygons)
        areas.append(result.area())

    assert abs(areas[1] - areas[2]) / areas[2] < 0.01


def test_union_contains_every_input():
    """The union must be at least as large as every input — this is the
    property that stops a canvas sized to it from clipping any member."""
    angles = np.linspace(-3, 3, 5)
    polygons = [_polygon_for_rotation(a) for a in angles]
    result = covering_polygon([SphericalPolygon(p) for p in polygons])
    for poly in polygons:
        assert result.area() >= poly.area()


# `intersect_polygons` is what a STACK uses: WBPP crops the integration to
# where the registered subframes overlap, so the intersection is the master's
# real extent rather than a conservative guess at it.

def test_intersection_smaller_than_all_inputs():
    angles = np.linspace(-3, 3, 5)
    polygons = [_polygon_for_rotation(a) for a in angles]
    result = intersect_polygons([SphericalPolygon(p) for p in polygons])
    for poly in polygons:
        assert result.area() < poly.area()


def test_intersection_and_cover_bracket_the_members():
    """The two strategies bracket every member: a stack's footprint can only
    shrink, a target's can only grow. Getting these backwards is what clipped
    the Heart's channels."""
    angles = np.linspace(-3, 3, 5)
    polygons = [_polygon_for_rotation(a) for a in angles]
    wrapped = [SphericalPolygon(p) for p in polygons]
    inner = intersect_polygons(wrapped).area()
    outer = covering_polygon(wrapped).area()
    for poly in polygons:
        assert inner <= poly.area() <= outer
