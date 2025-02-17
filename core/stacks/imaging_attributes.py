from dataclasses import dataclass
from typing import Any
import numpy as np
from beartype import beartype
from collections.abc import Iterable

from context import context
from metadata.classes import MetadataDict, FrameHash


@beartype
@dataclass(frozen=True)
class ImagingAttributes:
  image_type: str
  camera: str
  filter: str | None
  mean_temperature: float
  exposure_s: list[float]
  integration_s: float
  width_pixels: int
  height_pixels: int

  @classmethod
  def from_frames(cls,
                  frames: Iterable[FrameHash],
                  metadata: MetadataDict | None = None) -> 'ImagingAttributes':
    assert frames

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

    image_type = get_unique_quality('IMAGETYP')
    camera = get_unique_quality('INSTRUME')
    width_pixels = get_unique_quality('NAXIS1')
    height_pixels = get_unique_quality('NAXIS2')
    exposure_s = get_quality_set('EXPOSURE')
    filter = get_unique_quality('FILTER', allow_empty=True)

    integration_s = sum(
        i.header['EXPOSURE'] for i in metadata_list if 'EXPOSURE' in i.header)

    mean_temperature = np.mean([i.header['CCD-TEMP'] for i in metadata_list])

    return cls(image_type, camera, filter, mean_temperature, exposure_s,
               integration_s, width_pixels, height_pixels)
