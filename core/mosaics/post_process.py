import ray
from pathlib import Path
from astropy.time import Time

from pix import PixInsight
from pix.classes import ImageList
from pix.tasks import ChannelCombination, SpectrophotometricColorCalibration, ImageSolver, StarXTerminator, LinearFit
from context import context
from dependency_graph.astroflow_node import AstroFlowNode
from previews import QualityLevel
from mosaics.types import Mosaic


@PixInsight()
def _post_process_mosaic(images: ImageList, mosaic: Mosaic,
                         output_filename: Path):
  print(f'### _post_process_mosaic mosaic {mosaic} images {images}')

  r = next(iter(i for i in images if i.filter == 'R'))
  g = next(iter(i for i in images if i.filter == 'G'))
  b = next(iter(i for i in images if i.filter == 'B'))
  rgb = ChannelCombination([r, g, b], 'RGB')
  julian_date = int(Time(mosaic.oldest_date).jd)
  rgb = ImageSolver(rgb,
                    metadata_ra=mosaic.center[0],
                    metadata_dec=mosaic.center[1],
                    metadata_resolution=mosaic.image_scale / 3600.0,
                    metadata_observationTime=julian_date,
                    metadata_epoch=julian_date)
  rgb = SpectrophotometricColorCalibration(rgb)
  stars, _ = StarXTerminator(rgb)

  s = next(iter(i for i in images if i.filter == 'S'))
  h = next(iter(i for i in images if i.filter == 'H'))
  o = next(iter(i for i in images if i.filter == 'O'))
  _, s = StarXTerminator([s])
  _, h = StarXTerminator([h])
  _, o = StarXTerminator([o])
  s, o = LinearFit([s, o], h)
  narrowband = ChannelCombination([s, h, o], 'SHO')

  # TODO: Return list or tuples of images
  return narrowband


@AstroFlowNode(resources={'PixInsight': 1})
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
      Path(context.output_dir) / 'Mosaics' / mosaic.name /
      f'final_merged_{mosaic.stable_hash}_{filter}.fits' for filter in filters
  ]

  output_filename = Path(
      context.output_dir
  ) / 'Mosaics' / mosaic.name / f'final_merged_{mosaic.stable_hash}_post_processed.fits'

  _post_process_mosaic(images, mosaic, output_filename)
