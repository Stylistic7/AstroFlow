from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore
import astropy  # type: ignore

import units
from hashing import hash_list, hash_string, hash_list_int
from context import context
from stacks.simple_stack import SimpleStack, SimpleStacksDict
from stacks.imaging_attributes import ImagingAttributes
from stacks.state_attributes import AstroFlowStateAttributes
from stacks.astrometric_attributes import AstrometricAttributes
from stacks.chronology_attributes import ChronologyAttributes
from stacks.types import StackHash
from metadata.classes import MetadataDict, FrameHash
from catalogs import position_string
import geometry.footprints as footprints
from clustering.continuous import cluster_continuous
from plate_solving import PlateSolution


def _normalize_radians(x: float) -> float:
  if x < 0.0:
    return x + np.pi
  else:
    return x


@dataclass(frozen=True)
class CompositeStack(ImagingAttributes, AstrometricAttributes,
                     ChronologyAttributes, AstroFlowStateAttributes):
  stable_hash: str
  content_hash: str
  contents: SimpleStacksDict

  @classmethod
  def from_frames(
      cls,
      frames: list[FrameHash],
      stable_hash: StackHash | None = None,
      metadata: MetadataDict | None = None) -> Optional['CompositeStack']:
    if not frames:
      return None

    metadata = metadata or context.metadata
    assert metadata

    metadata_list = [i for h, i in metadata.items() if h in frames]

    plate_solution = metadata[frames[0]].plate_solution
    assert isinstance(plate_solution, PlateSolution)
    name = position_string.canonical_target_string(plate_solution.center)

    # Sort the light frames by observation date. The oldest image will be used
    # to compute the hash of the whole stack.
    sorted_metadata = sorted(metadata_list, key=lambda i: i.header['DATE-OBS'])
    oldest_image_metadata = sorted_metadata[0]

    assert oldest_image_metadata.plate_solution
    center = oldest_image_metadata.plate_solution.center

    image_scale = oldest_image_metadata.plate_solution.image_scale
    rotation_angle = _normalize_radians(
        oldest_image_metadata.plate_solution.rotation_angle)

    # TODO: Unify this WCS stuff and put it somewhere more general.
    # It's in target_identification and in mosaics/projection.
    footprint = SingleSphericalPolygon.from_wcs(
        astropy.io.fits.Header(oldest_image_metadata.plate_solution.header))
    if not footprint.is_clockwise():
      footprint = SingleSphericalPolygon(footprint.points[::-1])
    assert not SphericalPolygon.self_intersect(footprint.points)
    projected_footprints = footprints.all_projections(footprint)

    contents = SimpleStacksDict()
    exposures = [(i.content_hash, i.header['EXPOSURE'])
                 for i in metadata_list
                 if 'EXPOSURE' in i.header]
    for exposure_cluster in cluster_continuous(exposures, epsilon=0.01):
      exposure_frames, exposures = zip(*exposure_cluster)
      assert np.all(np.isclose(exposures, exposures[0])
                   ), f'exposures {exposures} are clustered, but not all close'

      runs = cluster_continuous(
          [(FrameHash(i.content_hash), i.header['DATE-OBS'])
           for i in sorted_metadata
           if i.content_hash in exposure_frames],
          epsilon=context.max_run_gap)

      for run in runs:
        if len(run) < context.min_frames_in_run:
          continue
        run_frames = [x[0] for x in run]
        new_stack = SimpleStack.from_frames(run_frames, metadata)
        contents[new_stack.content_hash] = new_stack

    if not contents:
      return None

    content_hash = hash_list(sorted(contents.keys()))

    chronology_attributes = ChronologyAttributes.from_frames(frames, metadata)
    imaging_attributes = ImagingAttributes.from_frames(frames, metadata)
    astrometric_attributes = AstrometricAttributes(
        name=name,
        center=list(center),
        image_scale=image_scale,
        rotation_angle=rotation_angle,
        footprint=footprint,
        projected_footprints=projected_footprints)

    if stable_hash is None:
      stable_hash = hash_string(
          f'RA {units.ra_to_hms(astrometric_attributes.center[0])}, '
          f'Dec {units.dec_to_dms(astrometric_attributes.center[1])}, '
          f'Scale {astrometric_attributes.image_scale:.2f}"/px, '
          f'PA {astrometric_attributes.rotation_angle:.2f}deg,'
          f'Width {imaging_attributes.width_pixels}, '
          f'Height {imaging_attributes.height_pixels}, '
          f'Filter {filter}')

    attributes: dict[str, Any] = asdict(chronology_attributes) | asdict(
        imaging_attributes) | asdict(astrometric_attributes) | {
            'contents': contents,
            'content_hash': content_hash,
            'stable_hash': stable_hash,
            'state': {},
        }

    return cls(**attributes)

  def all_frames(self) -> list[FrameHash]:
    all_frames = []
    for c in self.contents.values():
      all_frames.extend(c.frames)

    return all_frames

  def all_metadata(self) -> MetadataDict:
    all_metadata = {}
    for c in self.contents.values():
      all_metadata |= c.metadata

    return all_metadata

  def __str__(self) -> str:
    return (f'CompositeStack(Stable hash {self.stable_hash}, '
            f'Content hash {self.content_hash}, '
            f'Target {self.name}, '
            f'RA {units.ra_to_hms(self.center[0])}, '
            f'Dec {units.dec_to_dms(self.center[1])}, '
            f'Scale {self.image_scale:.2f}"/px, '
            f'PA {self.rotation_angle:.2f} deg, '
            f'Camera {self.camera}, '
            f'Exposure {self.exposure_s} s, '
            f'Integration {self.integration_s} s, '
            f'Filter {self.filter}, '
            f'Temp {self.mean_temperature:.02f} degC)')

  def __hash__(self):
    return hash_list_int([self.content_hash])

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, CompositeStack):
      return NotImplemented

    return (self.content_hash == other.content_hash) and (self.stable_hash
                                                          == other.stable_hash)


class CompositeStacksDict(dict[str, CompositeStack]):

  def __hash__(self):
    return hash_list_int([self[k].content_hash for k in sorted(self.keys())])

  def __str__(self):
    return f'SimpleStacksDict({len(self)})'


Stack = SimpleStack | CompositeStack
