from dataclasses import dataclass
from collections import namedtuple
from addict import Dict

import plate_solving


@dataclass(frozen=True)
class Panel:
  # TODO: Maybe remove these?
  mode: str = None
  mosaic_id: str = None
  filter: str = None

  tile_indices: list[int] = None

  # These define the input image.
  input_filename: str = None
  image_key: str = None
  image_hash: str = None

  # These are the stacks ids that this Panel represents.
  # TODO: Maybe call them `stack_stable_hash` and `stack_content_hash`.
  stable_hash: str = None
  content_hash: str = None

  # This defines where the Panel is in the tree of merge ops.
  merge_op_id: str = None

  # This defines the output.
  placed_filename: str = None


# TODO: Maybe use Artifact as a more strongly-typed Dict.
@dataclass(frozen=True)
class MergeOp:
  input_artifacts: list[Dict] = None
  output_artifact: Dict = None


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
