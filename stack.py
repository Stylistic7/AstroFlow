from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units
from hashing import hash_list

from typing import NewType, Optional

FrameHash = NewType('FrameHash', str)
StackHash = NewType('StackHash', str)


@dataclass
class Run:
  frames: list[FrameHash] = field(default_factory=list)
  stack_dependencies: list[StackHash] = field(default_factory=list)


@dataclass(order=True)
class Stack:
  runs: list[Run] = field(default_factory=list)
  target: Optional[str] = None
  camera: Optional[str] = None
  mean_temperature: Optional[float] = None
  exposure_s: Optional[int] = None
  width_pixels: Optional[int] = None
  height_pixels: Optional[int] = None
  center: Optional[list[float]] = None
  image_scale: Optional[float] = None
  rotation_angle: Optional[float] = None
  filter: Optional[str] = None
  footprint: Optional[SphericalPolygon] = None
  projected_footprints: Optional[dict[str, Polygon]] = None

  def footprint_polygon(self) -> SphericalPolygon:
    return SphericalPolygon(self.footprint)

  def hash(self) -> str:
    return hash_list(
        [tuple(r.frames + r.stack_dependencies) for r in self.runs])

  # TODO: It's important for development and debugging (at least) for stacks
  # composed entirely of bias frames or darks frames to have meaningful string
  # representations.
  def __str__(self) -> str:
    if (self.target and self.center and self.width_pixels and
        self.height_pixels and self.image_scale and self.filter):
      return (f'Stack({self.hash()}, {self.target}, '
              f'RA {units.ra_to_hms(self.center[0])}, '
              f'Dec {units.dec_to_dms(self.center[1])}, '
              f'{self.camera}, {self.image_scale:.2f}"/px, '
              f'{self.filter}')
    else:
      return (f'Stack({self.hash()})')
