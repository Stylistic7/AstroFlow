import numpy as np

from typing import overload

from geometry import Radians, PixelCoordinate
from geometry.projection import Projection
from geometry.projections.galactic import equatorial_to_galactic


class MollweideProjection(Projection):
  lon_0: Radians
  R: float

  def __init__(self, lon_0: Radians = 0, R: float = 45 * np.sqrt(2)):
    self.lon_0 = lon_0
    self.R = R

  @overload
  def _solve_theta(self, phi: float) -> float:
    ...

  @overload
  def _solve_theta(self, phi: np.ndarray) -> np.ndarray:
    ...

  def _solve_theta(self, phi: Radians) -> Radians:
    theta: Radians = np.zeros_like(phi) if isinstance(phi, np.ndarray) else 0

    for i in range(10):
      theta = theta - (2 * theta + np.sin(2 * theta) -
                       np.pi * np.sin(phi)) / (4 * np.cos(theta)**2.0)

    return theta

  @overload
  def project(self, lon: float, lat: float) -> tuple[float, float]:
    ...

  @overload
  def project(self, lon: np.ndarray,
              lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  def project(self, lon: Radians,
              lat: Radians) -> tuple[PixelCoordinate, PixelCoordinate]:
    theta = self._solve_theta(lat)

    x = self.R * (2 * np.sqrt(2) / np.pi) * (lon - self.lon_0) * np.cos(theta)
    y = self.R * np.sqrt(2) * np.sin(theta)

    return x, y

  @overload
  def unproject(self, x: float, y: float) -> tuple[float, float]:
    ...

  @overload
  def unproject(self, x: np.ndarray,
                y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  def unproject(self, x: PixelCoordinate,
                y: PixelCoordinate) -> tuple[Radians, Radians]:
    theta = np.arcsin(y / (self.R * np.sqrt(2)))
    lon = self.lon_0 + (x * np.pi) / (2 * self.R * np.sqrt(2) * np.cos(theta))
    lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)

    lon = np.where((lon >= 0) & (lon <= 2 * np.pi), lon, np.nan)
    lat = np.where((lat >= -np.pi) & (lat <= np.pi), lat, np.nan)

    return lon, lat

  def __repr__(self) -> str:
    return f'MollweideProjection(lon_0={self.lon_0} R={self.R})'


class MollweideGalacticProjection(Projection):
  lon_0: Radians
  R: float

  def __init__(self, lon_0: Radians = 0, R: float = 45 * np.sqrt(2)):
    self.proj = MollweideProjection(lon_0, R)

  @overload
  def project(self, lon: float, lat: float) -> tuple[float, float]:
    ...

  @overload
  def project(self, lon: np.ndarray,
              lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  def project(self, lon: Radians,
              lat: Radians) -> tuple[PixelCoordinate, PixelCoordinate]:
    assert isinstance(lon, float) and isinstance(lat, float)
    theta, phi = equatorial_to_galactic(lon, lat)
    return self.proj.project(theta, phi)

  @overload
  def unproject(self, x: float, y: float) -> tuple[float, float]:
    ...

  @overload
  def unproject(self, x: np.ndarray,
                y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  def unproject(self, x: PixelCoordinate,
                y: PixelCoordinate) -> tuple[Radians, Radians]:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f'MollweideGalacticProjection(lon_0={self.lon_0} R={self.R})'
