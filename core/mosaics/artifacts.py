from pathlib import Path

from context import context
from mosaics.types import Mosaic
from task.task import ContentHash
from previews.types import QualityLevel
from stacks.composite_stack import CompositeStack
from mosaics.types import TileIndices


# TODO: Use TIFF everywhere here.
def output_filename(mosaic: Mosaic, filter: str) -> Path:
  return Path(
      context.output_dir
  ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_{filter}.fits'


def merged_filename(mosaic: Mosaic, mode: QualityLevel, hash_0: ContentHash,
                    hash_1: ContentHash) -> Path:
  return Path(
      context.output_dir
  ) / 'Mosaics' / mosaic.name / f'merged_{mode}_{hash_0}_{hash_1}.fits'


def placed_filename(mosaic: Mosaic, mode: QualityLevel, stack: CompositeStack,
                    tile_indices: TileIndices, filter: str) -> Path:
  tile_indices_str = '_'.join(str(i) for i in tile_indices)
  return Path(context.output_dir) / 'Mosaics' / mosaic.name / Path(
      f'placed_image_{mode}_{mosaic.stable_hash}_{stack.stable_hash}_'
      f'{tile_indices_str}_{filter}.xisf')
