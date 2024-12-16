import os
from pathlib import Path

from artifacts import Artifact, FileArtifact
from context import context
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from catalogs import position_string
from stacks.types import StackHash


def output_filename(stack: SimpleStack | CompositeStack,
                    dest_path: str) -> Path:
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
            f'{stack.mean_temperature:.0f}C_{stack.exposure_s}s_'
            f'{stack.content_hash}.xisf'))
  elif isinstance(stack, SimpleStack) and stack.image_type == 'flat':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return Path(
        os.path.join(
            output_dir,
            f'master_flat_{stack.camera}_{stack.filter}_{stack.content_hash}.xisf'
        ))
  elif isinstance(stack, CompositeStack) and stack.image_type == 'light':
    output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    return Path(
        os.path.join(
            output_dir, f'master_light_{stack.target}_{ra_dec_string}_'
            f'{stack.camera}_{stack.image_scale:.02f}aspp_'
            f'{stack.rotation_angle:.2f}deg_{stack.filter}_{stack.content_hash}.xisf'
        ))

  assert False, f'unknown image_type {stack.image_type}'


def stack_files_artifacts(stack_id: StackHash) -> list[Artifact]:
  stack = context.stacks[stack_id]
  dest_filename = output_filename(stack, context.output_dir)
  return [FileArtifact(dest_filename, stack.content_hash)]
