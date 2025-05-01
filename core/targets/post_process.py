import ray
from typing import Any
import os
from pathlib import Path
from astropy.time import Time

from context import context
from catalogs import position_string
from pix.decorators import PixInsight
from pix.classes import ImageList
from pix.tasks import ChannelCombination, SpectrophotometricColorCalibration, ImageSolver, StarXTerminator, LinearFit, SaveImages
from file_io.types import XisfFilePath
from decorators import cached, measure_performance, create_thumbnail
from targets.types import Target
from stacks.artifacts import output_filename as stack_output_filename
from products import makes_image


# TODO: Find a way to disable these during tests.
@PixInsight(slots=[1])
def _post_process_target(images: ImageList, target: Target) -> ImageList:
  print(f'### _post_process_target target {target} images {images}')

  if 'RGB' in images:
    rgb = ChannelCombination(images.channels('RGB'))
    julian_date = int(Time(target.oldest_date).jd)
    rgb = ImageSolver(rgb,
                      metadata_ra=target.center[0],
                      metadata_dec=target.center[1],
                      metadata_resolution=target.image_scale / 3600.0,
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


# TODO: Targets are composed of multiple color channels, so this needs to be
# able to return multiple images.
def output_filename(target: Target) -> XisfFilePath:
  ra_dec_string = position_string.ra_dec_string(*target.center)
  return XisfFilePath(
      Path('Targets') / target.name / 'masters' /
      f'master_light_{target.name}_{ra_dec_string}_'
      f'{target.camera}_{target.image_scale:.02f}aspp_'
      f'{target.rotation_angle:.2f}deg_{target.filter}.xisf')


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@makes_image(Target).as_post_processed(output_filename)
@cached.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@measure_performance
@create_thumbnail
def post_process_target(target: Target) -> None:
  stacks = context.stacks
  images = [
      stack_output_filename(stacks[stack_id]).resolve_at(context.output_dir)
      for stack_id in target.stack_ids
  ]

  result = _post_process_target(images, target)
  print(f'post_process_target, target {target} returning {result}')
  # return XisfFilePath(result)
