from dataclasses import dataclass
from collections import namedtuple

from artifact import Artifact
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
  # TODO: Maybe remove these?
  mode: str = None
  mosaic_id: str = None
  filter: str = None

  tile_indices: list[int] = None

  # This define the input image.
  input_artifact: Artifact = None

  # These are the stacks ids that this Panel represents.
  # TODO: Maybe call them `stack_stable_hash` and `stack_content_hash`.
  stable_hash: str = None
  content_hash: str = None

  # This defines where the Panel is in the tree of merge ops.
  merge_op_id: str = None

  # This defines the output image.
  output_artifact: Artifact = None


@dataclass(frozen=True)
class MergeOp:
  input_artifacts: list[Artifact] = None
  output_artifact: Artifact = None


@dataclass
class MosaicDimensions:
  width: int = None
  height: int = None
  min_ra: float = None
  ra_span: float = None
  min_dec: float = None
  dec_span: float = None
  ra_resolution: float = None
  dec_resolution: float = None
  # world_positions: list = None
