import os
from pathlib import Path

from context import context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from catalogs import position_string
from file_io.types import XisfFilePath
from previews.types import QualityLevel


def output_filename(mode: QualityLevel,
                    stack: SimpleStack | CompositeStack) -> Path:

  def exposure_string(exposures):
    return '_'.join([f'{x}s' for x in exposures])

  if isinstance(stack, SimpleStack) and 'bias' in stack.image_type:
    output_dir = Path('Stacks') / 'Calibration' / mode
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_bias_{stack.camera}_'
            f'{stack.mean_temperature:.0f}C_{stack.stable_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and 'dark' in stack.image_type:
    output_dir = Path('Stacks') / 'Calibration' / mode
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_dark_{stack.camera}_'
            f'{stack.mean_temperature:.0f}C_{exposure_string(stack.exposure_s)}_'
            f'{stack.stable_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and 'flat' in stack.image_type:
    output_dir = Path('Stacks') / 'Calibration' / mode
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_flat_{stack.camera}_{stack.filter}_'
            f'{stack.stable_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and 'light' in stack.image_type:
    output_dir = Path('Runs') / mode
    return XisfFilePath(
        os.path.join(output_dir, f'master_light_run_{stack.stable_hash}.xisf'))
  elif isinstance(stack, CompositeStack) and 'light' in stack.image_type:
    output_dir = Path('Stacks') / stack.name / mode
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_light_{stack.name}_{ra_dec_string}_'
            f'{stack.camera}_{stack.image_scale:.02f}aspp_'
            f'{stack.rotation_angle:.2f}deg_{stack.filter}.xisf'))

  assert False, f'unknown image_type "{stack.image_type}"'
