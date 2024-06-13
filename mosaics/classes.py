from dataclasses import dataclass

import plate_solving


@dataclass(frozen=True)
class Panel:
  mosaic_id: str = None
  stable_hash: str = None
  content_hash: str = None
  mode: str = None
  filter: str = None
  image_hash: str = None
  placed_filename: str = None
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
