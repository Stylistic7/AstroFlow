import ray
from dataclasses import dataclass


@dataclass
class SpatialGroup:
  name: str
  light_frames: list[str]
  metadata_ref: ray.ObjectRef
