import os
from pathlib import Path
from astropy.time import Time
import shutil

from file_io.types import XisfFilePath
from pix import PixInsight
from pix.classes import ImageList
from pix.tasks import ChannelCombination, SpectrophotometricColorCalibration, ImageSolver, StarXTerminator, LinearFit, SaveImages
from context import context
from decorators.task_decorator import task
from previews.types import QualityLevel
from mosaics.types import Mosaic
from mosaics.artifacts import output_filename as mosaic_output_filename


@PixInsight(slots=[1])
def _post_process_mosaic(images: ImageList, mosaic: Mosaic,
                         output_filename: Path):
  print(f'### _post_process_mosaic mosaic {mosaic} images {images}')

  if 'RGB' in images:
    rgb = ChannelCombination(images.channels('RGB'))
    julian_date = int(Time(mosaic.oldest_date).jd)
    rgb = ImageSolver(rgb,
                      metadata_ra=mosaic.center[0],
                      metadata_dec=mosaic.center[1],
                      metadata_resolution=mosaic.image_scale / 3600.0,
                      metadata_observationTime=julian_date,
                      metadata_epoch=julian_date)
    rgb = SpectrophotometricColorCalibration(rgb)
    stars, _ = StarXTerminator(rgb)

  if 'SHO' in images:
    _, s = StarXTerminator(images.channels('s'))
    _, h = StarXTerminator(images.channels('h'))
    _, o = StarXTerminator(images.channels('o'))
    s, o = LinearFit([s, o], h)
    narrowband = ChannelCombination([s, h, o], 'SHO')

  # TODO: Supply multiple output filenames.
  SaveImages([narrowband], [output_filename])

  # TODO: Return list or tuples of images
  return []


@task.remote(resources={'PixInsight': 1})
def post_process_mosaic(mode: QualityLevel, mosaic: Mosaic) -> None:
  print(f'### Entering post_process_mosaic for mosaic {mosaic}')

  if mode != 'master':
    return None

  filters = set()
  for target_id in mosaic.target_ids:
    for stack_id in context.targets[target_id].stack_ids:
      filters.add(context.stacks[stack_id].filter)
  filters = tuple(filters)

  images = [mosaic_output_filename(mosaic, filter) for filter in filters]

  output_filename = XisfFilePath(
      context.output_dir
  ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_post_processed.xisf'

  if os.environ.get('MOCK_PIXINSIGHT'):
    shutil.copy(images[0], output_filename)
  else:
    _post_process_mosaic(images, mosaic, output_filename)

  return output_filename
