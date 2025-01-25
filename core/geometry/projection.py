import numpy as np
from abc import ABC, abstractmethod
from typing import overload

from geometry import PixelCoordinate, Radians


class Projection(ABC):

  @abstractmethod
  @overload
  def unproject(self, x: float, y: float) -> tuple[float, float]:
    ...

  @abstractmethod
  @overload
  def unproject(self, x: np.ndarray,
                y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  @abstractmethod
  def unproject(self, x: PixelCoordinate,
                y: PixelCoordinate) -> tuple[Radians, Radians]:
    pass

  @overload
  def project(self, lon: float, lat: float) -> tuple[float, float]:
    ...

  @overload
  def project(self, lon: np.ndarray,
              lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...

  @overload
  def project(self, lon: Radians,
              lat: Radians) -> tuple[PixelCoordinate, PixelCoordinate]:
    ...

  @abstractmethod
  def project(self, lon: Radians,
              lat: Radians) -> tuple[PixelCoordinate, PixelCoordinate]:
    pass

  # TODO: Create a Viewbox type.
  def sample_viewbox_lon_lat(self, viewbox: tuple[float, float, float, float],
                             num_samples_x: int) -> tuple[Radians, Radians]:
    x_min, y_min, x_span, y_span = viewbox
    resolution = x_span / num_samples_x
    num_samples_y = int(y_span / resolution)

    xx, yy = np.meshgrid(np.linspace(x_min, x_min + x_span, num_samples_x),
                         np.linspace(y_min, y_min + y_span, num_samples_y))

    return self.unproject(xx, yy)
