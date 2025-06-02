import os
from pathlib import Path

from context import context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from catalogs import position_string
from file_io.types import XisfFilePath


def output_filename(stack: SimpleStack | CompositeStack) -> Path:

  def exposure_string(exposures):
    return '_'.join([f'{x}s' for x in exposures])

  if isinstance(stack, SimpleStack) and 'bias' in stack.image_type:
    output_dir = Path('Stacks') / 'Calibration' / 'masters'
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_bias_{stack.camera}_'
            f'{stack.mean_temperature:.0f}C_{stack.content_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and 'dark' in stack.image_type:
    output_dir = Path('Stacks') / 'Calibration' / 'masters'
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_dark_{stack.camera}_'
            f'{stack.mean_temperature:.0f}C_{exposure_string(stack.exposure_s)}_'
            f'{stack.content_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and 'flat' in stack.image_type:
    output_dir = Path('Stacks') / 'Calibration' / 'masters'
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_flat_{stack.camera}_{stack.filter}_'
            f'{stack.content_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and 'light' in stack.image_type:
    output_dir = Path('Runs') / 'masters'
    return XisfFilePath(
        os.path.join(output_dir, f'master_light_run_{stack.content_hash}.xisf'))
  elif isinstance(stack, CompositeStack) and 'light' in stack.image_type:
    output_dir = Path('Stacks') / stack.name / 'masters'
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    # TOOD: Should this use the stack's content_hash or stable_hash? Using the
    # content hash means keeping a lot of older (obsolete) copies of the stack
    # as it changes. On the other hand, the stack artifact is small (not much
    # bigger than a subframe) but requires a lot of computation to generate.
    return XisfFilePath(
        os.path.join(
            output_dir, f'master_light_{stack.name}_{ra_dec_string}_'
            f'{stack.camera}_{stack.image_scale:.02f}aspp_'
            f'{stack.rotation_angle:.2f}deg_{stack.filter}.xisf'))

  assert False, f'unknown image_type "{stack.image_type}"'
