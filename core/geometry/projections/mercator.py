import numpy as np

from typing import overload

from geometry.projection import Projection
from geometry import Radians, PixelCoordinate


class MercatorProjection(Projection):
    lon_0: Radians

    def __init__(self, lon_0: Radians = 0) -> None:
        self.lon_0 = lon_0

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
        x = lon - self.lon_0
        y = np.log(np.tan((np.pi / 4) + (lat / 2)))

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
        lon = self.lon_0 + x
        lat = 2 * np.arctan(np.exp(y)) - (np.pi / 2)

        return lon, lat

    def __repr__(self) -> str:
        return f"MercatorProjection(lon_0={self.lon_0})"
