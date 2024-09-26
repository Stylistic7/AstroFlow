from dataclasses import dataclass, fields
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units
import numpy as np
import datetime
from typing import NewType, Any
from hashing import hash_list
from metadata import MetadataDict
from context import context

FrameHash = NewType('FrameHash', str)
StackHash = NewType('StackHash', str)


@dataclass
class ImagingAttributes:
  image_type: str
  camera: str
  filter: str | None
  mean_temperature: float
  exposure_s: list[int]
  integration_s: int
  width_pixels: int
  height_pixels: int
  oldest_date: datetime.datetime
  newest_date: datetime.datetime

  def __init__(self,
               frames: list[FrameHash],
               metadata: MetadataDict | None = None) -> None:
    metadata = metadata or context.metadata

    metadata_list = [i for i in metadata.values() if i.content_hash in frames]

    def get_unique_quality(key: str, allow_empty: bool = False) -> Any:
      values = list(set(
          i.header[key] for i in metadata_list if key in i.header))
      if allow_empty and len(values) == 0:
        return None
      assert len(values) == 1, f'Key {key} had non-unique values {values}'
      return values[0]

    def get_quality_set(key: str) -> list[Any]:
      return list(set(i.header[key] for i in metadata_list if key in i.header))

    self.image_type = get_unique_quality('IMAGETYP')
    self.camera = get_unique_quality('INSTRUME')
    self.width_pixels = get_unique_quality('NAXIS1')
    self.height_pixels = get_unique_quality('NAXIS2')
    self.exposure_s = get_quality_set('EXPOSURE')
    self.filter = get_unique_quality('FILTER', allow_empty=True)

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

  # This hash defines the specific area of the sky this stack covers. It
  # includes the center (RA, Dec), the rotation angle, the image scale, and
  # the image width and height, in pixels. It is not defined for stacks that
  # do not contain light frames.
  def __init__(self, target: str, center: tuple[float,
                                                float], image_scale: float,
               rotation_angle: float, footprint: SphericalPolygon,
               projected_footprints: dict[str, Polygon]) -> str:
    self.target = target
    self.center = center
    self.image_scale = image_scale
    self.rotation_angle = rotation_angle
    self.footprint = footprint
    self.projected_footprints = projected_footprints


@dataclass
class SimpleStack(ImagingAttributes, AstroFlowStateAttributes):
  content_hash: str
  frames: list[FrameHash]
  dependencies: dict[str, StackHash]

  def __init__(self,
               frames: list[FrameHash],
               metadata: MetadataDict | None = None):
    self.frames = frames
    self.content_hash = hash_list(frames)
    self.dependencies = {}
    self.state = {}

    metadata = metadata or context.metadata
    imaging_attributes = ImagingAttributes(frames, metadata)
    for field in fields(ImagingAttributes):
      setattr(self, field.name, getattr(imaging_attributes, field.name))

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

  def __init__(self, contents: list[StackHash],
               imaging_attributes: ImagingAttributes,
               astrometric_attributes: AstrometricAttributes):
    for field in fields(ImagingAttributes):
      setattr(self, field.name, getattr(imaging_attributes, field.name))

    for field in fields(AstrometricAttributes):
      setattr(self, field.name, getattr(astrometric_attributes, field.name))

    self.contents = contents
    self.content_hash = hash_list(contents)
    self.stable_hash = (f'RA {units.ra_to_hms(self.center[0])}, '
                        f'Dec {units.dec_to_dms(self.center[1])}, '
                        f'Scale {self.image_scale:.2f}"/px, '
                        f'PA {self.rotation_angle:.2f}deg,'
                        f'Width {self.width_pixels}, '
                        f'Height {self.height_pixels}, '
                        f'Filter {filter}')

  def all_frames(self, stacks: 'StacksDict') -> list[FrameHash]:
    stacks = stacks or context.stacks

    all_frames = []
    for substack_id in self.contents:
      substack = stacks[substack_id]
      assert isinstance(substack, SimpleStack)
      all_frames.extend(substack.frames)

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
StacksDict = dict[str, Stack]
