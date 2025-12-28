from pathlib import Path
from shapely import box  # type: ignore

from context import context
from file_io.types import XisfFilePath, TiffFilePath, WithSidecar
from mosaics.types import Mosaic
from protocols import ContentHash
from previews.types import QualityLevel
from customization.types import CustomizedStack
from mosaics.types import Mosaic, Viewbox
from file_io.generic import get_content_hash


def final_filename(mode: QualityLevel, mosaic: Mosaic, filter: str) -> Path:
  if mode == QualityLevel.SUBFRAME:
    return WithSidecar(
        TiffFilePath(
            Path('Mosaics') / mode / mosaic.name /
            f'final_{mosaic.stable_hash}_{filter}.tiff'))
  else:
    return WithSidecar(
        XisfFilePath(
            Path('Mosaics') / mode / mosaic.name /
            f'final_{mosaic.stable_hash}_{filter}.xisf'))


def tile_filename(mosaic: Mosaic, filter: str) -> TiffFilePath:
  filename = Path(final_filename(QualityLevel.MASTER, mosaic, filter))
  return TiffFilePath(
      filename.with_stem(filename.stem + '_autostretched').with_suffix('.tiff'))


def merged_filename(mode: QualityLevel, mosaic: Mosaic, image_0: WithSidecar,
                    image_1: WithSidecar) -> Path:
  hash_0 = get_content_hash(image_0)
  hash_1 = get_content_hash(image_1)

  if mode == QualityLevel.SUBFRAME:
    return WithSidecar(
        TiffFilePath(
            Path('Mosaics') / mode / mosaic.name /
            f'merged_{hash_0}_{hash_1}.tiff'))
  else:
    return WithSidecar(
        XisfFilePath(
            Path('Mosaics') / mode / mosaic.name /
            f'merged_{hash_0}_{hash_1}.xisf'))


def placed_filename(mode: QualityLevel, mosaic: Mosaic, stack: CustomizedStack,
                    filter: str, viewbox: Viewbox, mask: box) -> Path:
  return TiffFilePath(
      Path('Mosaics') / mode / mosaic.name /
      f'placed_image_{mosaic.stable_hash}_{stack.stable_hash}_'
      f'{filter}.tiff')
