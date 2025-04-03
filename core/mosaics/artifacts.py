from pathlib import Path

from context import context
from mosaics.types import Mosaic


def output_filename(mosaic: Mosaic, filter: str):
  return Path(
      context.output_dir
  ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_{filter}.fits'
