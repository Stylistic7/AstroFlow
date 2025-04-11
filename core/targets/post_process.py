import ray
from typing import Any
import os
from pathlib import Path

from catalogs import position_string
from artifacts import Artifact
from pix.decorators import PixInsight
from pix.classes import ImageList
from targets.types import TargetHash
from file_io.types import XisfFilePath
from decorators import cached, makes_image, measure_performance, create_thumbnail
from targets.types import Target
from previews.types import QualityLevel


# TODO: Find a way to disable these during tests.
@PixInsight(slots=[1])
def _post_process_target(images: ImageList) -> ImageList:
  # print('### Entering _post_process_target')
  # aligned = pix.StarAlignment(images)
  # print('### pix.StarAlignment returned', aligned)
  # # TODO: Run ChannelCombination, ImageSolver, SPCC before doing
  # # StarXTerminator.
  # # TODO: ImageSolver must write the target's coordinates into the RGB image
  # # so ImageSolver won't hang.
  # result = pix.StarXTerminator(aligned)
  # print('### pix.StarXTerminator returned', result)
  # return result
  return images


# TODO: Targets are composed of multiple color channels, so this needs to be
# able to return multiple images.
def output_filename(target: Target) -> XisfFilePath:
  output_dir = Path('Targets') / target.name / 'masters'
  ra_dec_string = position_string.ra_dec_string(*target.center)
  return XisfFilePath(
      os.path.join(
          output_dir, f'master_light_{target.name}_{ra_dec_string}_'
          f'{target.camera}_{target.image_scale:.02f}aspp_'
          f'{target.rotation_angle:.2f}deg_{target.filter}.xisf'))


# TODO: Hash the script used to do the post-processing, and do not repeat
# the post-processing when a file with that hash already exists.
@makes_image(Target).as_post_processed(output_filename)
@cached.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@measure_performance
@create_thumbnail
def post_process_target(target_id: TargetHash,
                        images: ImageList) -> list[Artifact]:
  result = _post_process_target(images)
  print(f'post_process_target, target_id {target_id} returning {result}')
  return XisfFilePath(result)
