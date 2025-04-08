from dataclasses import dataclass
from beartype import beartype
import numpy as np
import astropy  # type: ignore
from typing import Any, Sequence

from spherical_geometry.polygon import SphericalPolygon, SingleSphericalPolygon  # type: ignore

from context import context
from metadata.classes import MetadataDict, FrameHash
from catalogs import position_string


def _normalize_radians(x: float) -> float:
  if x < 0.0:
    return float(x + np.pi)
  else:
    return float(x)


@beartype
@dataclass(frozen=True)
class AstrometricAttributes:
  name: str
  center: list[float]
  image_scale: float
  rotation_angle: float
  footprint: SingleSphericalPolygon

  @classmethod
  def from_frames(
      cls,
      frames: Sequence[FrameHash],
      metadata: MetadataDict | None = None) -> 'AstrometricAttributes':
    assert frames

    metadata = metadata or context.metadata
    metadata_list = [i for i in metadata.values() if i.content_hash in frames]

    sorted_metadata = sorted(metadata_list, key=lambda i: i.header['DATE-OBS'])
    oldest_image_metadata = sorted_metadata[0]

    assert oldest_image_metadata.plate_solution
    center = list(oldest_image_metadata.plate_solution.center)
    name = position_string.canonical_target_string(center)

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

    return cls(name, center, image_scale, rotation_angle, footprint)
