import ray
from dataclasses import dataclass
from beartype import beartype


@beartype
@dataclass(frozen=True)
class SpatialGroup:
    name: str
    light_frames: tuple[str, ...]
    metadata_ref: ray.ObjectRef
