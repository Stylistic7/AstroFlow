import os
from pathlib import Path
from astropy.time import Time
import shutil

from file_io.types import XisfFilePath
from pix import PixInsight
from pix.classes import ImageList
from pix.tasks import ChannelCombination, SpectrophotometricColorCalibration, ImageSolver, StarXTerminator, LinearFit, SaveImages
from context import context
from decorators import cached, create_thumbnail, makes_image, measure_performance
from previews.types import QualityLevel
from mosaics.types import Mosaic
from mosaics.artifacts import post_processed_filename, merged_color_channel


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


@makes_image(Mosaic).as_post_processed(post_processed_filename)
@cached.remote(resources={'PixInsight': 1})
@measure_performance
@create_thumbnail
def post_process_mosaic(mode: QualityLevel, mosaic: Mosaic) -> None:
  print(f'### Entering post_process_mosaic for mosaic {mosaic}')

  if mode != 'master':
    return None

  filters = set()
  for target_id in mosaic.target_ids:
    for stack_id in context.targets[target_id].stack_ids:
      filters.add(context.stacks[stack_id].filter)
  filters = tuple(filters)

  images = [
      context.output_dir / merged_color_channel(mosaic, filter)
      for filter in filters
  ]

  for image in images:
    print(
        f'### post_process_mosaic image {image} exists {os.path.exists(image)}')

  output_filename = context.output_dir / post_processed_filename(mosaic, None)
  print(f'### post_process_mosaic output_filename {output_filename}')

  if os.environ.get('MOCK_PIXINSIGHT'):
    shutil.copy(images[0], output_filename)
  else:
    _post_process_mosaic(images, mosaic, output_filename)

  return output_filename
