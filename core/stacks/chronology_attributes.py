from dataclasses import dataclass
import datetime
from typing import Any
import numpy as np

from context import context
from metadata.classes import MetadataDict, FrameHash


@dataclass(frozen=True)
class ChronologyAttributes:
  oldest_date: datetime.datetime
  newest_date: datetime.datetime

  @classmethod
  def from_frames(
      cls,
      frames: list[FrameHash],
      metadata: MetadataDict | None = None) -> 'ChronologyAttributes':
    assert frames

    metadata = metadata or context.metadata
    metadata_list = [i for i in metadata.values() if i.content_hash in frames]

    sorted_metadata = sorted(metadata_list, key=lambda i: i.header['DATE-OBS'])
    oldest_date = sorted_metadata[0].header['DATE-OBS']
    newest_date = sorted_metadata[-1].header['DATE-OBS']

    return cls(oldest_date, newest_date)
