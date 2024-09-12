from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units
import numpy as np
import datetime
from typing import NewType, Optional
from hashing import hash_list
from context import context
import time
from copy import copy

FrameHash = NewType('FrameHash', str)
StackHash = NewType('StackHash', str)


@dataclass
class ImagingAttributes:
  image_type: Optional[str] = None
  camera: Optional[str] = None
  filter: Optional[str] = None
  mean_temperature: Optional[float] = None
  exposure_s: Optional[int] = None
  integration_s: Optional[int] = None
  width_pixels: Optional[int] = None
  height_pixels: Optional[int] = None
  oldest_date: Optional[datetime.datetime] = None
  newest_date: Optional[datetime.datetime] = None

  def update_metadata(self) -> None:
    all_frames = self.all_frames()
    self.content_hash = hash_list(all_frames)

    metadata = [
        i for i in context.metadata.values() if i.content_hash in all_frames
    ]

    def get_quality(quality, allow_multiple=False):
      # TODO: What should we do with frames that don't have a FILTER keyword?
      qualities = list(
          set(i.header[quality] for i in metadata if quality in i.header))
      if len(qualities) == 0:
        return None

      if type(qualities[0]) is float and np.all(
          np.isclose(qualities, qualities[0])):
        qualities = {qualities[0]}

      if allow_multiple:
        return list(qualities)
      else:
        assert len(qualities) < 2, f'multiple {quality} qualities {qualities}'
        return next(iter(qualities)) if len(qualities) else None

    # allow_multiple_exposures = (self.stable_hash != None)

    self.image_type = get_quality('IMAGETYP')
    self.camera = get_quality('INSTRUME')
    self.width_pixels = get_quality('NAXIS1')
    self.height_pixels = get_quality('NAXIS2')
    self.exposure_s = get_quality('EXPOSURE', allow_multiple=True)
    self.filter = get_quality('FILTER')

    self.integration_s = sum(
        i.header['EXPOSURE'] for i in metadata if 'EXPOSURE' in i.header)

    self.mean_temperature = np.mean([i.header['CCD-TEMP'] for i in metadata])

    sorted_metadata = sorted(metadata, key=lambda i: i.header['DATE-OBS'])
    self.oldest_date = sorted_metadata[0].header['DATE-OBS']
    self.newest_date = sorted_metadata[-1].header['DATE-OBS']


# TODO: Store the best light frame during subframe selection; it will be used
# in many places, e.g. thumbnailing, and eliminate a lot of unnecessary calls
# to get_metadata().
@dataclass
class AstroFlowStateAttributes:
  state: dict = field(default_factory=dict)


@dataclass
class AstrometricAttributes:
  target: Optional[str] = None
  center: Optional[list[float]] = None
  image_scale: Optional[float] = None
  rotation_angle: Optional[float] = None
  footprint: Optional[SphericalPolygon] = None
  projected_footprints: Optional[dict[str, Polygon]] = None
  selection_criterion: Optional[tuple[str, float]] = None


@dataclass
class SimpleStack(ImagingAttributes, AstroFlowStateAttributes):
  content_hash: Optional[str] = None
  frames: list[FrameHash] = field(default_factory=list)
  dependencies: dict[str, StackHash] = field(default_factory=dict)

  def all_frames(self, include_dependencies=False):
    stacks = context.stacks
    all_frames = copy(self.frames)

    # TODO: This doesn't handle user settings that will override
    # dependencies.
    if include_dependencies:
      for dependency_stack_id in self.dependencies.values():
        all_frames.extend(stacks[dependency_stack_id].frames)

    return all_frames

  def __str__(self):
    return (f'Stack(Content hash {self.content_hash}, '
            f'Type {self.image_type}, '
            f'Camera {self.camera}, '
            f'Exposure {self.exposure_s}, '
            f'Integration {self.integration_s} s, '
            f'Filter {self.filter}, '
            f'Temp {self.mean_temperature:.02f} degC)')

  def __eq__(self, other):
    if not isinstance(other, SimpleStack):
      return False

    return (self.content_hash == other.content_hash)


@dataclass
class CompositeStack(ImagingAttributes, AstrometricAttributes,
                     AstroFlowStateAttributes):
  stable_hash: Optional[str] = None
  content_hash: Optional[str] = None
  contents: list[StackHash] = field(default_factory=list)

  def all_frames(self, include_dependencies=False) -> list[FrameHash]:
    stacks = context.stacks
    all_frames = []
    for substack_id in self.contents:
      substack = stacks[substack_id]
      all_frames.extend(substack.all_frames(include_dependencies))

    return all_frames

  def __str__(self):
    return (f'Stack(Stable hash {self.stable_hash}, '
            f'Content hash {self.content_hash}, '
            f'Target {self.target}, '
            f'RA {units.ra_to_hms(self.center[0])}, '
            f'Dec {units.dec_to_dms(self.center[1])}, '
            f'Scale {self.image_scale:.2f}"/px, '
            f'PA {self.rotation_angle:.2f} deg, '
            f'Camera {self.camera}, '
            f'Exposure {self.exposure_s} s, '
            f'Integration {self.integration_s} s, '
            f'Filter {self.filter}, '
            f'Temp {self.mean_temperature:.02f} degC)')

  def __eq__(self, other):
    if not isinstance(other, CompositeStack):
      return False

    return (self.content_hash == other.content_hash) and (self.stable_hash
                                                          == other.stable_hash)


Stack = SimpleStack | CompositeStack
