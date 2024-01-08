from dataclasses import dataclass

import plate_solving


@dataclass
class Panel:
  stack_id: str = None
  image_filename: str = None
  plate_solution: plate_solving.PlateSolution = None
  tile_indices: list[int] = None


@dataclass
class MergeOp:
  image_filenames: list[str] = None
  output_filename: str = None


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
