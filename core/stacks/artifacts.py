import os
from pathlib import Path

from context import context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from catalogs import position_string


def output_filename(stack: SimpleStack | CompositeStack,
                    dest_path: str | None = None) -> Path:
  if dest_path is None:
    dest_path = context.output_dir

  def exposure_string(exposures):
    return '_'.join([f'{x}s' for x in exposures])

  if isinstance(stack, SimpleStack) and stack.image_type == 'bias':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return Path(
        os.path.join(
            output_dir, f'master_bias_{stack.camera}_'
            f'{stack.mean_temperature:.0f}C_{stack.content_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and stack.image_type == 'dark':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return Path(
        os.path.join(
            output_dir, f'master_dark_{stack.camera}_'
            f'{stack.mean_temperature:.0f}C_{exposure_string(stack.exposure_s)}_'
            f'{stack.content_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and stack.image_type == 'flat':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return Path(
        os.path.join(
            output_dir, f'master_flat_{stack.camera}_{stack.filter}_'
            f'{stack.content_hash}.xisf'))
  elif isinstance(stack, CompositeStack) and stack.image_type == 'light':
    output_dir = os.path.join(dest_path, 'Stacks', stack.name, 'masters')
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    return Path(
        os.path.join(
            output_dir, f'master_light_{stack.name}_{ra_dec_string}_'
            f'{stack.camera}_{stack.image_scale:.02f}aspp_'
            f'{stack.rotation_angle:.2f}deg_{stack.filter}_'
            f'{stack.content_hash}.xisf'))

  assert False, f'unknown image_type {stack.image_type}'
