from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units
import numpy as np
import datetime
from typing import NewType, Any
from hashing import hash_list
from context import context
from metadata import ImageMetadata
import time
from copy import copy

FrameHash = NewType('FrameHash', str)
StackHash = NewType('StackHash', str)

Stacks = dict[str, Stack]


@dataclass
class ImagingAttributes:
  image_type: str
  camera: str
  filter: str
  mean_temperature: float
  exposure_s: list[int]
  integration_s: int
  width_pixels: int
  height_pixels: int
  oldest_date: datetime.datetime
  newest_date: datetime.datetime

  @abstractmethod
  def all_frames(self, stacks: Stacks) -> list[FrameHash]:
    pass

  # TODO: The `stacks` argument here is used to select between `grouped_stacks`,
  # which are the raw stacks computed from the files on disk, before any user
  # input, and `stacks`, after quarantining and subframe selection. This is not
  # very elegant, and could be improved.
  def update_metadata(self,
                      stacks: Stacks | None = None,
                      metadata: dict[str, ImageMetadata] | None = None) -> None:
    stacks = stacks or {}
    metadata = metadata or context.metadata

    all_frames = self.all_frames(stacks)
    self.content_hash = hash_list(all_frames)

    metadata_list = [
        i for i in metadata.values() if i.content_hash in all_frames
    ]

    def get_unique_quality(key: str) -> Any:
      values = list(set(
          i.header[key] for i in metadata_list if key in i.header))
      assert len(values) == 1
      return values[0]

    def get_quality_set(key: str) -> list[Any]:
      return list(set(i.header[key] for i in metadata_list if key in i.header))

    self.image_type = get_unique_quality('IMAGETYP')
    self.camera = get_unique_quality('INSTRUME')
    self.width_pixels = get_unique_quality('NAXIS1')
    self.height_pixels = get_unique_quality('NAXIS2')
    self.exposure_s = get_quality_set('EXPOSURE')
    self.filter = get_unique_quality('FILTER')

    self.integration_s = sum(
        i.header['EXPOSURE'] for i in metadata_list if 'EXPOSURE' in i.header)

    self.mean_temperature = np.mean(
        [i.header['CCD-TEMP'] for i in metadata_list])

    sorted_metadata = sorted(metadata_list, key=lambda i: i.header['DATE-OBS'])
    self.oldest_date = sorted_metadata[0].header['DATE-OBS']
    self.newest_date = sorted_metadata[-1].header['DATE-OBS']


# TODO: Store the best light frame during subframe selection; it will be used
# in many places, e.g. thumbnailing, and eliminate a lot of unnecessary calls
# to get_metadata().
@dataclass
class AstroFlowStateAttributes:
  state: dict


@dataclass
class AstrometricAttributes:
  target: str
  center: list[float]
  image_scale: float
  rotation_angle: float
  footprint: SphericalPolygon
  projected_footprints: dict[str, Polygon]
  selection_criterion: tuple[str, float]


@dataclass
class SimpleStack(ImagingAttributes, AstroFlowStateAttributes):
  content_hash: str
  frames: list[FrameHash]
  dependencies: dict[str, StackHash]

  def all_frames(self,
                 stacks: Stacks,
                 include_dependencies: bool = False) -> list[FrameHash]:
    if stacks is None:
      stacks = context.stacks

    assert self.frames
    all_frames = copy(self.frames)

    # TODO: This doesn't handle user settings that will override
    # dependencies.
    if include_dependencies:
      for dependency_stack_id in self.dependencies.values():
        dependency_stack = stacks[dependency_stack_id]
        assert isinstance(dependency_stack, SimpleStack)
        all_frames.extend(dependency_stack.frames)

    return all_frames

  def __str__(self) -> str:
    return (f'Stack(Content hash {self.content_hash}, '
            f'Type {self.image_type}, '
            f'Camera {self.camera}, '
            f'Exposure {self.exposure_s}, '
            f'Integration {self.integration_s} s, '
            f'Filter {self.filter}, '
            f'Temp {self.mean_temperature:.02f} degC)')

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, SimpleStack):
      return NotImplemented

    return (self.content_hash == other.content_hash)


@dataclass
class CompositeStack(ImagingAttributes, AstrometricAttributes,
                     AstroFlowStateAttributes):
  stable_hash: str
  content_hash: str
  contents: list[StackHash]

  def all_frames(self,
                 stacks: Stacks,
                 include_dependencies: bool = False) -> list[FrameHash]:
    stacks = stacks or context.stacks

    all_frames = []
    for substack_id in self.contents:
      substack = stacks[substack_id]
      assert isinstance(substack, SimpleStack)
      all_frames.extend(substack.all_frames(stacks, include_dependencies))

    return all_frames

  def __str__(self) -> str:
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

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, CompositeStack):
      return NotImplemented

    return (self.content_hash == other.content_hash) and (self.stable_hash
                                                          == other.stable_hash)


Stack = SimpleStack | CompositeStack
