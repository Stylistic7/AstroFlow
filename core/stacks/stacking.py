import tempfile
import os
from pathlib import Path
import errno
import os
import shutil
import astropy.io.fits as pyfits  # type: ignore
from typing import Any

import file_io
from file_io.types import XisfFilePath
from file_io.generic import update_header
from stacks.artifacts import output_filename
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import Stack, CompositeStack
from customization.types import CustomizedStack
from context import context
from pix import PixInsight, ImageList
from pix.tasks import WeightedBatchPreprocessing
from metadata.classes import MetadataDict
from decorators import cached, makes_image, measure_performance, create_thumbnail


def _symlink_frames(filenames: list[str], dest_path: str) -> None:
  for filename in filenames:
    assert filename
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

    if frame.plate_solution:
      with pyfits.open(str(dest_filename), mode='update', mmap=False) as fits:
        fits[0].header['RA'] = frame.plate_solution.header['CRVAL1']
        fits[0].header['DEC'] = frame.plate_solution.header['CRVAL2']
        for k, v in frame.plate_solution.header.items():
          fits[0].header[k] = v


@PixInsight()
def _run_wbpp(unused: ImageList, *args: Any, **kwargs: Any) -> Any:
  return WeightedBatchPreprocessing(unused, *args, **kwargs)


# TODO: This function should call _setup_simple_stack for its leaves.
def _setup_composite_stack(stack: Stack,
                           tmp_dir: str) -> tuple[str | None, int]:
  metadata = context.metadata
  calibration_stacks = context.calibration_stacks
  print(f'### calibration_stacks keys {calibration_stacks.keys()}')

  first_filename = None

  num_frames = 0
  for i, substack in enumerate(stack.contents.values()):
    filenames = [
        i.filename for h, i in metadata.items() if h in substack.frames
    ]
    assert len(substack.frames) == len(filenames)

    if first_filename is None:
      first_filename = filenames[0]

    run_dir = os.path.join(tmp_dir, f'RUN_{i}')
    os.makedirs(run_dir, exist_ok=True)

    run_metadata = {h: i for h, i in metadata.items() if h in substack.frames}
    _copy_frames_with_plate_solution(run_metadata, run_dir)
    num_frames += len(substack.frames)

    # Add any dependent stacks (e.g. calibration stacks).
    dependendency_filenames = [
        output_filename(calibration_stacks[d]).resolve_at(context.output_dir)
        for d in substack.dependencies.values()
    ]
    _symlink_frames(dependendency_filenames, run_dir)

  return first_filename, num_frames


def _setup_simple_stack(stack: Stack, tmp_dir: str) -> tuple[str | None, int]:
  filenames = [stack.metadata[h].filename for h in stack.frames]
  _symlink_frames(filenames, tmp_dir)

  return filenames[0], len(filenames)


# TOOD: Implement a @singledispatch version of this that distinguishes leaf
# from composite stacks.
#
# TODO: When a stack fails, keep note of it so it won't be retried for the
# exact same inputs again.
@makes_image(CompositeStack).as_master(output_filename)
@cached.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@measure_performance
@create_thumbnail
def stack_files(stack: Stack) -> XisfFilePath:
  dest_filename = stack_files.filename_fn(stack).resolve_at(context.output_dir)
  print(f'### Entering stack_files for {stack} dest_filename {dest_filename}')

  metadata = {'IMAGETYP': f'Master {stack.image_type}'}

  # TODO: This is kind of dangerous, because the output filename only depends
  # on the stack's own content hash -- the hash used by AstroFlowNode is more
  # complete and includes this function's code, for example.
  if context.dangerously_accept_existing_files and os.path.exists(
      dest_filename):
    print(f'### Dangerously skipping stacking because '
          f'file {dest_filename} already exists')
    return dest_filename

  os.makedirs(os.path.dirname(dest_filename), exist_ok=True)

  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
    if type(stack) is SimpleStack:
      first_filename, num_frames = _setup_simple_stack(stack, tmp_dir)
    elif type(stack) is CustomizedStack:
      first_filename, num_frames = _setup_composite_stack(stack, tmp_dir)

    if num_frames < 3:
      print(f'### Cannot stack {stack} because it contains only '
            f'{num_frames} selected frames')
      return dest_filename

    if os.environ.get('MOCK_PIXINSIGHT'):
      assert first_filename is not None
      file_io.conversion.convert_fits_to_xisf(first_filename,
                                              dest_filename,
                                              add_noise=True,
                                              metadata=metadata)
      print(f'### Mocked stacking, converted filename {first_filename} to '
            f'filename {dest_filename} with metadata {metadata}')

      return dest_filename

    print(f'### Running _run_wbpp for stack {stack}')
    _run_wbpp([], dir=tmp_dir, outputDirectory=tmp_dir)

    master_dir = os.path.join(tmp_dir, 'master')
    if not os.path.exists(master_dir):
      raise Exception("WBPP did not produce an output master dir")

    print('WBPP master dir contains:')
    for f in os.listdir(master_dir):
      print(f'  {f}')

    wbpp_masters = list(
        str(f.absolute()) for f in Path(master_dir).rglob("master*.xisf"))
    wbpp_masters.sort(key=lambda x: os.path.getmtime(x))
    filename = wbpp_masters[-1]
    print(f'last modified WBPP master {filename} -> {dest_filename}')
    shutil.copy(filename, dest_filename)
    update_header(dest_filename, metadata)

    return dest_filename
