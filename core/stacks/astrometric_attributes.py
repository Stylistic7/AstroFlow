from dataclasses import dataclass

from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore


@dataclass(frozen=True)
class AstrometricAttributes:
  target: str
  center: list[float]
  image_scale: float
  rotation_angle: float
  footprint: SphericalPolygon
  projected_footprints: dict[str, Polygon]
