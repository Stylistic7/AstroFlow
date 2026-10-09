"""Stereographic (STG) projection centred on a mosaic, with a position angle.

WHY (2026-10-09). Multi-panel mosaics were projected with Mercator, built with
its central meridian at its default, RA 0h. Its x is `lon - lon_0` with no
wrap, so a mosaic straddling RA 0h (Sh2-171) put its panels ~360 deg apart and
sized a canvas of 586,150 px — 141 GB per float32 array. Every merge of it was
killed by macOS at 250-380 GB of compressed memory, eleven times in one night,
each kill restarting the whole graph. Mercator is also the wrong family for
the sky: its scale grows as sec(dec), so high-declination mosaics stretch.

Stereographic is conformal like Mercator (stars stay round everywhere) but
centred, so it has no wrap and no pole: every position is measured from the
mosaic's own centre. Its scale grows as sec^2(c/2) with distance c from the
centre — 1.04 at 22.5 deg (the edge of the 45 deg mosaic span cap), against
1.17 for the gnomonic plane — and near the centre the two are identical, so a
single field looks as the camera saw it either way.

Units and rotation follow `GnomonicProjection` exactly: radians, unit sphere
(scale 1 at the centre, so a pixel is `image_scale` there), and `rotation`
turns the plane axes so a field at that position angle comes out axis-aligned.
"""
from __future__ import annotations

from typing import overload

import numpy as np

from geometry import PixelCoordinate, Radians
from geometry.projection import Projection


class StereographicProjection(Projection):
    """STG projection centred at `(lon_0, lat_0)`, rotated by `rotation`
    (all radians)."""

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

    def project(
        self, lon: Radians, lat: Radians
    ) -> tuple[PixelCoordinate, PixelCoordinate]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        sin_lat0, cos_lat0 = np.sin(self.lat_0), np.cos(self.lat_0)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        d_lon = lon - self.lon_0          # sin/cos make the RA wrap irrelevant
        cos_dlon, sin_dlon = np.cos(d_lon), np.sin(d_lon)

        # cos of the angular distance from the centre; -1 is the antipode,
        # the one point STG cannot represent (never inside a mosaic).
        cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * cos_dlon
        k = 2.0 / np.maximum(1.0 + cos_c, 1e-12)

        x = k * cos_lat * sin_dlon
        y = k * (cos_lat0 * sin_lat - sin_lat0 * cos_lat * cos_dlon)

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
        safe = np.where(rho == 0, 1.0, rho)   # the centre itself: (lon_0, lat_0)
        c = 2.0 * np.arctan(rho / 2.0)
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
        return (f"StereographicProjection(lon_0={self.lon_0}, lat_0={self.lat_0}, "
                f"rotation={self.rotation})")
