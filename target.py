import numpy as np
from dataclasses import dataclass, field
import functools
from shapely.geometry import Polygon

import plate_solving
import spatial_association

import matplotlib.pyplot as plt


@dataclass
class Target:
  # TODO: This currently just means "good" or not. Elaborate.
  status: bool = False

  name: str = None

  stacks: list[str] = field(default_factory=list)

  def __repr__(self):
    return (f'Target(name={self.name}, {len(self.stacks)} stacks)')