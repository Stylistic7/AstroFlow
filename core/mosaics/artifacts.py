from pathlib import Path

from context import context
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from mosaics.types import Mosaic
from protocols import ContentHash
from previews.types import QualityLevel
from stacks.composite_stack import CompositeStack
from mosaics.types import TileIndices


def merged_color_channel(mosaic: Mosaic, filter: str) -> Path:
  return WithSidecar(
      XisfFilePath(
          Path('Mosaics') / mosaic.name /
          f'final_merged_{mosaic.stable_hash}_{filter}.xisf'))


def tile_filename(mosaic: Mosaic, filter: str) -> TiffFilePath:
  filename = Path(merged_color_channel(mosaic, filter))
  return TiffFilePath(
      filename.with_stem(filename.stem + '_autostretched').with_suffix('.tiff'))


def merged_filename(mosaic: Mosaic, mode: QualityLevel, hash_0: ContentHash,
                    hash_1: ContentHash) -> Path:
  if mode == QualityLevel.SUBFRAME:
    return WithSidecar(
        TiffFilePath(
            Path('Mosaics') / mosaic.name /
            f'merged_{mode}_{hash_0}_{hash_1}.tiff'))
  else:
    return WithSidecar(
        XisfFilePath(
            Path('Mosaics') / mosaic.name /
            f'merged_{mode}_{hash_0}_{hash_1}.xisf'))


def placed_filename(mosaic: Mosaic, mode: QualityLevel, stack: CompositeStack,
                    tile_indices: TileIndices, filter: str) -> Path:
  tile_indices_str = '_'.join(str(i) for i in tile_indices)
  return TiffFilePath(
      Path('Mosaics') / mosaic.name /
      f'placed_image_{mode}_{mosaic.stable_hash}_{stack.stable_hash}_'
      f'{tile_indices_str}_{filter}.tiff')


def post_processed_filename(mosaic: Mosaic, filter: str):
  return XisfFilePath(
      Path('Mosaics') / mosaic.name /
      f'final_merged_{mosaic.stable_hash}_post_processed.xisf')
