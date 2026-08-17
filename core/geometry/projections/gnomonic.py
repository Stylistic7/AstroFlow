"""Gnomonic (TAN) projection — the tangent plane, with a position angle.

WHY (2026-08-17). Targets were projected with Mercator, which is north-up by
construction and has local scale `sec(delta)`. Both properties are wrong for a
single field:

  * NO ROTATION. Mercator has no position-angle parameter, so a target shot at
    PA 88.8 deg lands rotated in a north-up canvas, with wedge corners and a
    bounding box far larger than the data.
  * A TAPER. The scale varies across the field in declination. For the Heart
    nebula at delta ~ +61.5 deg spanning ~2.5 deg:
        cos(60.25) / cos(62.75) = 0.4962 / 0.4571 = 1.086
    i.e. one end 8.6% wider than the other — which is exactly the taper seen
    in the rendered images.

TAN is what every astronomical package uses for a single field: it is the
projection a telescope actually forms on a flat sensor, so a frame reprojected
into a TAN plane tangent at its own centre, rotated to its own position angle,
comes back looking like the frame that was shot. Distortion over a degree or two
is negligible (it grows as tan of the angular distance from the tangent point);
it is unusable for whole-sky work, which is why mosaics keep Mercator.

Sign convention: `rotation` rotates the tangent-plane axes so that a field whose
`rotation_angle` is theta comes out axis-aligned. `project` then `unproject`
round-trips to sub-pixel over a field of view — the property the tests pin,
because a sign error here does not raise, it silently misplaces pixels.
"""
from __future__ import annotations

import numpy as np
from typing import overload

from geometry import PixelCoordinate, Radians
from geometry.projection import Projection


class GnomonicProjection(Projection):
    """TAN projection tangent at `(lon_0, lat_0)`, rotated by `rotation`.

    All angles are RADIANS — including `rotation`, which matches
    `AstrometricAttributes.rotation_angle` (a value of 1.55 is 88.8 degrees,
    not 1.55 degrees; that unit has been misread before).
    """

    lon_0: Radians
    lat_0: Radians
    rotation: Radians

    def __init__(self, lon_0: Radians = 0, lat_0: Radians = 0,
                 rotation: Radians = 0) -> None:
        self.lon_0 = lon_0
        self.lat_0 = lat_0
        self.rotation = rotation

    @overload
    def project(self, lon: float, lat: float) -> tuple[float, float]: ...

    @overload
    def project(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def project(self, lon: Radians, lat: Radians) -> tuple[Radians, Radians]: ...

    def project(
        self, lon: Radians, lat: Radians
    ) -> tuple[PixelCoordinate, PixelCoordinate]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        sin_lat0, cos_lat0 = np.sin(self.lat_0), np.cos(self.lat_0)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        d_lon = lon - self.lon_0
        cos_dlon, sin_dlon = np.cos(d_lon), np.sin(d_lon)

        # cos of the angular distance from the tangent point. It goes to zero at
        # 90 deg and negative beyond — the far hemisphere, which TAN cannot
        # represent. Clipping keeps the arithmetic finite; callers never project
        # a field that large (a mosaic spanning that much uses Mercator).
        cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon
        cos_c = np.where(np.abs(cos_c) < 1e-12, 1e-12, cos_c)

        x = cos_lat * sin_dlon / cos_c
        y = (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon) / cos_c

        if self.rotation:
            cos_r, sin_r = np.cos(self.rotation), np.sin(self.rotation)
            x, y = x * cos_r + y * sin_r, -x * sin_r + y * cos_r
        return x, y

    @overload
    def unproject(self, x: float, y: float) -> tuple[float, float]: ...

    @overload
    def unproject(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def unproject(
        self, x: PixelCoordinate, y: PixelCoordinate
    ) -> tuple[Radians, Radians]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.rotation:
            cos_r, sin_r = np.cos(self.rotation), np.sin(self.rotation)
            x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r

        rho = np.hypot(x, y)
        # At the tangent point rho is 0 and `y / rho` is undefined; the limit is
        # simply (lon_0, lat_0), so special-case it rather than emit a nan.
        safe = np.where(rho == 0, 1.0, rho)
        c = np.arctan(rho)
        sin_c, cos_c = np.sin(c), np.cos(c)
        sin_lat0, cos_lat0 = np.sin(self.lat_0), np.cos(self.lat_0)

        lat = np.arcsin(
            np.clip(cos_c * sin_lat0 + y * sin_c * cos_lat0 / safe, -1.0, 1.0))
        lon = self.lon_0 + np.arctan2(
            x * sin_c, safe * cos_lat0 * cos_c - y * sin_lat0 * sin_c)

        lat = np.where(rho == 0, self.lat_0, lat)
        lon = np.where(rho == 0, self.lon_0, lon)
        return lon, lat

    def __repr__(self) -> str:
        return (f"GnomonicProjection(lon_0={self.lon_0}, lat_0={self.lat_0}, "
                f"rotation={self.rotation})")
