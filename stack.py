from dataclasses import dataclass
from spherical_geometry.polygon import SphericalPolygon
from shapely.geometry import Polygon
import units


@dataclass(order=True)
class Stack:
  id: str = None
  content: list = None
  target: str = None
  camera: str = None
  width_pixels: int = None
  height_pixels: int = None
  center: list = None
  image_scale: float = None
  rotation_angle: float = None
  filter: str = None
  footprint: SphericalPolygon = None
  projected_footprints: dict[str, Polygon] = None

  def footprint_polygon(self):
    return SphericalPolygon(self.footprint)

  def __str__(self):
    return (f'Stack({self.target}, {self.id}, {self.status}, '
            f'RA {units.ra_to_hms(self.center[0])}, '
            f'Dec {units.dec_to_dms(self.center[1])}, '
            f'{self.width_pixels}x{self.height_pixels}, '
            f'{self.image_scale:.2f}"/px, '
            f'{self.filter}, {len(self.light_frames)} light frames)')
