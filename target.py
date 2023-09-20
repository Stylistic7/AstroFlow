import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon

import plate_solving
import spatial_association

import matplotlib.pyplot as plt


@dataclass
class Target:
  id: str = None

  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  name: str = None

  light_frames: list[str] = field(default_factory=list)
  stacks: list[str] = field(default_factory=list)

  # def __post_init__(self):
  #   self.id = hash(self.light_frames)

  def __eq__(self, other):
    return self.light_frames == other.light_frames

  def __repr__(self):
    return (f'Target(name={self.name}, {len(self.stacks)} stacks)')