from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units

import datetime
from typing import NewType, Optional

FrameHash = NewType('FrameHash', str)
RunHash = NewType('RunHash', str)
StackHash = NewType('StackHash', str)


@dataclass
class Run:
  content_hash: RunHash | None = None
  frames: list[FrameHash] = field(default_factory=list)
  num_frames: Optional[int] = None
  type: Optional[str] = None
  exposure_s: Optional[int] = None
  stack_dependencies: list[StackHash] = field(default_factory=list)


# TODO: Split this into light frame stacks (which have preceding stacks), and
# calibration stacks (which only have calibration frames).
@dataclass(order=True)
class Stack:
  geometric_hash: str | None = None
  content_hash: str | None = None

  oldest_date = Optional[datetime.datetime]
  newest_date = Optional[datetime.datetime]

  type: Optional[str] = None
  runs: list[Run] = field(default_factory=list)
  target: Optional[str] = None
  camera: Optional[str] = None
  mean_temperature: Optional[float] = None
  num_frames: Optional[int] = None
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

  def __eq__(self, other):
    if not isinstance(other, Stack):
      return False

    return (self.geometric_hash == other.geometric_hash and
            self.content_hash == other.content_hash)

  # TODO: It's important for development and debugging (at least) for stacks
  # composed entirely of bias frames or darks frames to have meaningful string
  # representations.
  def __str__(self) -> str:
    if (self.target and self.center and self.rotation_angle and self.camera and
        self.image_scale and self.filter):
      return (f'Stack(Geometric hash {self.geometric_hash}, '
              f'Content hash {self.content_hash}, '
              f'Target {self.target}, '
              f'RA {units.ra_to_hms(self.center[0])}, '
              f'Dec {units.dec_to_dms(self.center[1])}, '
              f'Scale {self.image_scale:.2f}"/px, '
              f'PA {self.rotation_angle:.2f}deg, '
              f'Camera {self.camera}, Filter {self.filter})')
    else:
      return (f'Stack(Content hash {self.content_hash})')
