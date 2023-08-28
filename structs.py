import numpy as np
from dataclasses import dataclass, field
import functools

# from compact_repr import compact_repr
import plate_solving
import spatial_association


@dataclass
class Stack:
  target: str = None
  rig: str = None
  filter: str = None

  light_frames: list[str] = field(default_factory=list)
  dark_frames: list[str] = field(default_factory=list)
  bias_frames: list[str] = field(default_factory=list)
  flat_frames: list[str] = field(default_factory=list)

  flat_frame_associations: list = field(default_factory=list)

  camera_configs = None

  master: str = None
  post_processed: str = None

  def center(self):
    plate_solutions = plate_solving.solve_fields(self.light_frames)
    centers = [c for c, f, _ in plate_solutions]
    return np.mean(centers, axis=0)

  def footprint(self):
    plate_solutions = plate_solving.solve_fields(self.light_frames)
    footprints = [f for _, f, _ in plate_solutions]
    return functools.reduce(spatial_association.intersection, footprints, None)

  def __repr__(self):
    return (
        f'Stack(target={self.target}, rig={self.rig}, filter={self.filter}, '
        f'{len(self.light_frames)} light frames)')