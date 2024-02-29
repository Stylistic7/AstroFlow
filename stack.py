from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units
from hashing import hash_list, hash_string
import file_io

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

  selection_criterion: Optional[tuple[str, float]] = None

  def footprint_polygon(self) -> SphericalPolygon:
    return SphericalPolygon(self.footprint)

  # This hash defines the specific area of the sky this stack covers. It
  # includes the center (RA, Dec), the rotation angle, the image scale, and
  # the image width and height, in pixels. It is not defined for stacks that
  # do not contain light frames.
  def geometric_hash(self) -> str:
    if (self.center and self.image_scale and self.rotation_angle and
        self.width_pixels and self.height_pixels):
      return hash_string(
          f'RA {units.ra_to_hms(self.center[0])}, '
          f'Dec {units.dec_to_dms(self.center[1])}, '
          f'Scale {self.image_scale:.2f}"/px, '
          f'PA {self.rotation_angle:.2f}deg,'
          f'Width {self.width_pixels}, Height {self.height_pixels}')

  # This hash includes runs of frames, e.g. light frames, plus all other stacks
  # that this stack depends on, e.g. stacks of bias and dark frames.
  def content_hash(self) -> str:
    return hash_list(
        [tuple(r.frames + r.stack_dependencies) for r in self.runs])

  def hash(self) -> str:
    geometric_hash = self.geometric_hash()
    if geometric_hash:
      return geometric_hash

    return self.content_hash()

  # TODO: It's important for development and debugging (at least) for stacks
  # composed entirely of bias frames or darks frames to have meaningful string
  # representations.
  def __str__(self) -> str:
    if (self.target and self.center and self.rotation_angle and self.camera and
        self.image_scale and self.filter):
      return (f'Stack(Geometric hash {self.geometric_hash()}, '
              f'Content hash {self.content_hash()}, '
              f'Target {self.target}, '
              f'RA {units.ra_to_hms(self.center[0])}, '
              f'Dec {units.dec_to_dms(self.center[1])}, '
              f'Scale {self.image_scale:.2f}"/px, '
              f'PA {self.rotation_angle:.2f}deg,'
              f'Camera {self.camera}, Filter {self.filter})')
    else:
      return (f'Stack(Content hash {self.content_hash()})')
