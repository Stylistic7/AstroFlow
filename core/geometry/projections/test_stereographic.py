"""The stereographic projection mosaics are placed in. As with the gnomonic
plane, a bug here does not raise: placement inverse-maps every canvas pixel
through `unproject`, so the inverse must be exact, and a mosaic straddling
RA 0h must stay compact (the Mercator canvas went 360 deg wide, 2026-10-09)."""
import numpy as np
import pytest

from geometry.projections.gnomonic import GnomonicProjection
from geometry.projections.stereographic import StereographicProjection


@pytest.mark.parametrize("lat0", [0.0, 45.0, 61.5, -30.0, 85.0, 90.0])
@pytest.mark.parametrize("rotation", [0.0, 30.0, 88.8, -60.0])
def test_round_trip_over_a_wide_mosaic(lat0, rotation):
    p = StereographicProjection(np.radians(10.0), np.radians(lat0), np.radians(rotation))
    rng = np.random.default_rng(1)
    lon = np.radians(10.0) + rng.uniform(-0.4, 0.4, 500)      # ±23 deg
    lat = np.clip(np.radians(lat0) + rng.uniform(-0.4, 0.4, 500), -np.pi / 2, np.pi / 2)
    lon2, lat2 = p.unproject(*p.project(lon, lat))
    # angular separation on the sphere (haversine: exact for tiny angles), arcsec
    h = np.sin((lat2 - lat) / 2) ** 2 + np.cos(lat) * np.cos(lat2) * np.sin((lon2 - lon) / 2) ** 2
    sep = np.degrees(2 * np.arcsin(np.sqrt(h))) * 3600
    assert sep.max() < 1e-4


def test_a_mosaic_straddling_ra_zero_stays_compact():
    """Panels at RA 359 and 1 deg are 2 deg apart, not 358."""
    p = StereographicProjection(0.0, np.radians(60.0))
    x, _ = p.project(np.radians(np.array([359.0, 1.0])), np.radians(np.array([60.0, 60.0])))
    assert abs(np.degrees(x[1] - x[0])) < 1.1      # 2 deg * cos(60) ≈ 1


def test_scale_is_one_at_the_centre_and_mild_at_22_degrees():
    p = StereographicProjection(0.0, 0.0)
    eps = 1e-6
    x, _ = p.project(np.array([0.0, eps]), np.array([0.0, 0.0]))
    assert abs((x[1] - x[0]) / eps - 1.0) < 1e-6
    c = np.radians(22.5)
    x, _ = p.project(np.array([c, c + eps]), np.array([0.0, 0.0]))
    assert abs((x[1] - x[0]) / eps - 1 / np.cos(c / 2) ** 2) < 1e-4      # sec^2(c/2) ≈ 1.04


def test_matches_the_gnomonic_plane_near_the_centre():
    s, g = StereographicProjection(1.0, 0.7, 0.3), GnomonicProjection(1.0, 0.7, 0.3)
    lon, lat = 1.0 + np.array([0.001, -0.002]), 0.7 + np.array([0.0015, 0.0005])
    np.testing.assert_allclose(s.project(lon, lat), g.project(lon, lat), atol=1e-8)
