import ray
import tempfile
import os
from pathlib import Path
import errno
import os
import shutil
import astropy.io.fits as pyfits  # type: ignore
from typing import Any
from functools import singledispatch

import file_io
from file_io.types import XisfFilePath
from file_io.generic import update_header
from stacks.artifacts import output_filename
from stacks.simple_stack import SimpleStack, SimpleStacksDict
from stacks.composite_stack import Stack, CompositeStack
from customization.types import CustomizedStack
from context import context
from pix import PixInsight, ImageList
from pix.tasks import WeightedBatchPreprocessing
from metadata.classes import MetadataDict
from decorators import cached, measure_performance, create_thumbnail
from products import makes_image


def _symlink_frames(filenames: list[str], dest_path: str) -> None:
  for filename in filenames:
    assert filename, f"bad filename {filename}"
    if not os.path.exists(filename):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'symlinking {filename} -> {dest_filename}')
    os.symlink(filename, dest_filename)


def _copy_frames_with_plate_solution(metadata: MetadataDict,
                                     dest_path: str) -> None:
  for frame in metadata.values():
    filename = frame.filename
    assert filename
    if not os.path.exists(filename):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'copying {filename} -> {dest_filename}')
    shutil.copyfile(filename, dest_filename)

    with pyfits.open(str(dest_filename), mode='update', mmap=False) as fits:
      if 'invalid' in fits[0].header.get('BAYERPAT', '').lower():
        del fits[0].header['BAYERPAT']

      if frame.plate_solution:
        fits[0].header['RA'] = frame.plate_solution.header['CRVAL1']
        fits[0].header['DEC'] = frame.plate_solution.header['CRVAL2']
        for k, v in frame.plate_solution.header.items():
          fits[0].header[k] = v


@PixInsight()
def _run_wbpp(unused: ImageList, *args: Any, **kwargs: Any) -> Any:
  return WeightedBatchPreprocessing(unused, *args, **kwargs)


def _setup_simple_stack(stack: SimpleStack, metadata: MetadataDict,
                        calibration_stacks: SimpleStacksDict,
                        tmp_dir: str) -> None:
  filenames = [i.filename for h, i in metadata.items() if h in stack.frames]
  assert len(stack.frames) == len(filenames)

  run_metadata = {h: i for h, i in metadata.items() if h in stack.frames}
  _copy_frames_with_plate_solution(run_metadata, tmp_dir)

  def get_dependency_filename(d):
    if 'stack_id' in d:
      return output_filename(calibration_stacks[d['stack_id']]).resolve_at(
          context.output_dir)
    elif 'filename' in d:
      return d['filename']

  # Add any dependent stacks (e.g. calibration stacks).
  dependendency_filenames = [
      get_dependency_filename(d) for d in stack.dependencies.values()
  ]
  _symlink_frames(dependendency_filenames, tmp_dir)

  return filenames[0]


def _setup_composite_stack(stack: CompositeStack, metadata: MetadataDict,
                           calibration_stacks: SimpleStacksDict,
                           tmp_dir: str) -> tuple[str | None, int]:
  first_filename = None

  for i, substack in enumerate(stack.contents.values()):
    run_dir = os.path.join(tmp_dir, f'RUN_{i}')
    os.makedirs(run_dir, exist_ok=True)
    result = _setup_simple_stack(substack, metadata, calibration_stacks,
                                 run_dir)
    first_filename = first_filename or result

  return first_filename


def _mock_stacking(src_filename: str, dest_filename: str,
                   metadata: dict[str, str]) -> None:
  assert src_filename is not None
  file_io.conversion.convert_fits_to_xisf(src_filename,
                                          dest_filename,
                                          add_noise=True,
                                          metadata=metadata)
  print(f'### Mocked stacking, converted filename {src_filename} to '
        f'filename {dest_filename} with metadata {metadata}')
  sidecar = Path(src_filename).with_suffix('.pkl')
  if sidecar.exists():
    dest_sidecar = dest_filename.with_suffix('.pkl')
    shutil.copy(sidecar, dest_sidecar)
    print(f'### Copied plate-solving sidecar {sidecar} to {dest_sidecar}')

  return XisfFilePath(dest_filename)


def _copy_wbpp_result(tmp_dir: str, dest_filename: str, metadata: dict[str,
                                                                       str]):
  # The 'master' dir indicates success.
  master_dir = os.path.join(tmp_dir, 'master')
  if not os.path.exists(master_dir):
    raise Exception("WBPP did not produce an output master dir")

  print('WBPP master dir contains:')
  for f in os.listdir(master_dir):
    print(f'  {f}')

  # Copy the last file written, and update its metadata.
  wbpp_masters = list(
      str(f.absolute()) for f in Path(master_dir).rglob("master*.xisf"))
  wbpp_masters.sort(key=lambda x: os.path.getmtime(x))
  filename = wbpp_masters[-1]
  print(f'last modified WBPP master {filename} -> {dest_filename}')
  shutil.copy(filename, dest_filename)
  update_header(dest_filename, metadata)

  return XisfFilePath(dest_filename)


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def _stack_with_pixinsight(stack: Stack, metadata: dict[str, str]) -> Path:
  dest_filename = stack_files.filename_fn(stack).resolve_at(context.output_dir)
  os.makedirs(os.path.dirname(dest_filename), exist_ok=True)

  calibration_stacks = context.calibration_stacks
  print(f'### calibration_stacks keys {calibration_stacks.keys()}')

  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
    setup_fn = _setup_composite_stack if type(stack) in [
        CompositeStack, CustomizedStack
    ] else _setup_simple_stack
    first_filename = setup_fn(stack, context.metadata, calibration_stacks,
                              tmp_dir)

    # Maybe mock PixInsight (in tests).
    if context.get('mock_pixinsight', False):
      return _mock_stacking(first_filename, dest_filename, metadata)

    print(f'### Running _run_wbpp for stack {stack}')
    _run_wbpp([], dir=tmp_dir, outputDirectory=tmp_dir)
    return _copy_wbpp_result(tmp_dir, dest_filename, metadata)


@makes_image(CompositeStack).as_master(output_filename)
@cached.remote
@measure_performance
@create_thumbnail
def stack_files(stack: Stack) -> XisfFilePath:
  # Determine the output file, and the metadata it should have.
  dest_filename = stack_files.filename_fn(stack).resolve_at(context.output_dir)
  os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
  print(f'### Entering stack_files for {stack} dest_filename {dest_filename}')

  metadata = {
      'IMAGETYP': f'Master {stack.image_type}',
      'STBLHASH': stack.stable_hash
  }

  # Maybe skip stacking if the output file already exists.
  # TODO: This is kind of dangerous, because the output filename only depends
  # on the stack's own content hash -- the hash used by AstroFlowNode is more
  # complete and includes this function's code, for example.
  if context.dangerously_accept_existing_files and os.path.exists(
      dest_filename):
    print(f'### Dangerously skipping stacking because '
          f'file {dest_filename} already exists')
    update_header(dest_filename, metadata)
    return dest_filename

  # Run WBPP.
  return ray.get(_stack_with_pixinsight.remote(stack, metadata))
