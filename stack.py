from dataclasses import dataclass, field
from spherical_geometry.polygon import SphericalPolygon  # type: ignore
from shapely.geometry import Polygon  # type: ignore
import units
import numpy as np
import datetime
from typing import NewType, Optional
from hashing import hash_list

FrameHash = NewType('FrameHash', str)
StackHash = NewType('StackHash', str)


# TODO: Maybe split the geometric stuff out into an abstract class. That way,
# Stack will really have two subclasses, FrameStack and ContainerStack? These
# are not good names. They could be two totally separate classes, but it's
# really nice to look them up in the same dictionary.
#
# Stacks made of frames have all of the camera info, but no geometric info.
# Stacks made of lower-level Stacks have all the camera info, PLUS the
# geometric info.
@dataclass
class Stack:
  content_hash: Optional[str] = None
  frames: list[FrameHash] = field(default_factory=list)
  dependencies: dict[str, StackHash] = field(default_factory=dict)
  image_type: Optional[str] = None
  oldest_date: Optional[datetime.datetime] = None
  newest_date: Optional[datetime.datetime] = None
  camera: Optional[str] = None
  filter: Optional[str] = None
  mean_temperature: Optional[float] = None
  exposure_s: Optional[int] = None
  integration_s: Optional[int] = None
  width_pixels: Optional[int] = None
  height_pixels: Optional[int] = None

  stable_hash: Optional[str] = None
  contents: list[StackHash] = field(default_factory=list)
  target: Optional[str] = None
  center: Optional[list[float]] = None
  image_scale: Optional[float] = None
  rotation_angle: Optional[float] = None
  footprint: Optional[SphericalPolygon] = None
  projected_footprints: Optional[dict[str, Polygon]] = None
  selection_criterion: Optional[tuple[str, float]] = None

  def all_frames(self, stacks) -> list[str]:
    return self.frames + [f for s in self.contents for f in stacks[s].frames]

  def update_metadata(self, fits_headers, stacks=None) -> None:
    if stacks == None:
      stacks = {}

    if self.stable_hash:
      assert self.contents
      assert not self.frames
      assert not self.dependencies
    else:
      assert self.frames
      assert not self.contents

    all_frames = self.all_frames(stacks)
    self.content_hash = hash_list(all_frames)

    def get_quality(quality, allow_multiple=False):
      qualities = list(
          set(fits_headers[f].get(quality, None) for f in all_frames))
      if type(qualities[0]) is float and np.all(
          np.isclose(qualities, qualities[0])):
        qualities = {qualities[0]}

      if allow_multiple:
        return list(qualities)
      else:
        assert len(qualities) < 2, f'multiple {quality} qualities {qualities}'
        return next(iter(qualities)) if len(qualities) else None

    allow_multiple_exposures = (self.stable_hash != None)

    self.image_type = get_quality('IMAGETYP')
    self.camera = get_quality('INSTRUME')
    self.width_pixels = get_quality('NAXIS1')
    self.height_pixels = get_quality('NAXIS2')
    self.exposure_s = get_quality('EXPOSURE', allow_multiple_exposures)
    self.filter = get_quality('FILTER')

    self.integration_s = sum([fits_headers[f]['EXPOSURE'] for f in all_frames])

    self.mean_temperature = np.mean(
        [fits_headers[f]['CCD-TEMP'] for f in all_frames])

    all_dates = sorted(fits_headers[f]['DATE-OBS'] for f in all_frames)
    self.oldest_date = all_dates[0]
    self.newest_date = all_dates[-1]

    # TODO: Update stable hash here, too.

  def __eq__(self, other):
    if not isinstance(other, Stack):
      return False

    return (self.content_hash == other.content_hash) and (self.stable_hash
                                                          == other.stable_hash)

  def __str__(self):
    if self.stable_hash:
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
    else:
      return (f'Stack(Content hash {self.content_hash}, '
              f'Camera {self.camera}, '
              f'Exposure {self.exposure_s}, '
              f'Integration {self.integration_s} s, '
              f'Filter {self.filter}, '
              f'Temp {self.mean_temperature:.02f} degC)')

  # TODO: Add a valid() method, and assert on it.
