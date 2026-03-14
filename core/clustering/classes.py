import ray
from dataclasses import dataclass
from beartype import beartype


@beartype
@dataclass
class SpatialGroup:
    name: str
    light_frames: list[str]
    metadata_ref: ray.ObjectRef
