from dataclasses import dataclass
from collections import namedtuple

from artifacts import Artifact
from geometry.projection import Projection


@dataclass(frozen=True)
class Viewbox:
  projection: Projection
  x_min: float
  x_span: float
  y_min: float
  y_span: float


@dataclass(frozen=True)
class Panel:
  tile_indices: tuple[int, int]

  # This define the input image.
  input_artifact: Artifact

  content_hash: str


@dataclass(frozen=True)
class MergeOp:
  input_artifacts: list[Artifact]
  output_artifact: Artifact


@dataclass
class MosaicDimensions:
  width: int
  height: int
  min_ra: float
  ra_span: float
  min_dec: float
  dec_span: float
  ra_resolution: float
  dec_resolution: float
