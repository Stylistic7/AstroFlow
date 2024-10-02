import tempfile
import os
from pathlib import Path
import numpy as np
import ray
import errno
import os
import shutil
import astropy.io.fits as pyfits  # type: ignore
from typing import Any

from stacks.artifacts import output_filename, stack_files_artifacts
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from stacks.types import StackHash, FrameHash
from artifacts.decorators import saves_artifacts
from artifacts import FileArtifact, Artifact
from context import context
from pix import PixInsight, FilenameList
from pix.tasks import WeightedBatchPreprocessing
import file_io
from metadata.classes import MetadataDict
from metadata.actor import get_metadata
from profiling import track_performance


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
def _run_wbpp(unused: FilenameList, *args: Any, **kwargs: Any) -> Any:
  return WeightedBatchPreprocessing(unused, *args, **kwargs)


# TODO: This function should call _setup_simple_stack for its leaves.
def _setup_composite_stack(stack_id: StackHash,
                           tmp_dir: str) -> tuple[str | None, int]:
  quarantined_stacks = context.quarantined_stacks or set()
  if stack_id in quarantined_stacks:
    return None, 0

  metadata = context.metadata
  stacks = context.stacks
  stack = stacks[stack_id]
  hashes = stack.all_frames()

  first_filename = None

  num_frames = 0
  for i, substack in enumerate(stack.contents.values()):
    filenames = [i.filename for h, i in metadata.items() if h in hashes]
    assert len(hashes) == len(filenames)

    if first_filename is None:
      first_filename = filenames[0]

    run_dir = os.path.join(tmp_dir, f'RUN_{i}')
    os.makedirs(run_dir, exist_ok=True)

    run_metadata = {h: i for h, i in metadata.items() if h in hashes}
    _copy_frames_with_plate_solution(run_metadata, run_dir)
    num_frames += len(hashes)

    # Add any dependent stacks (e.g. calibration stacks).
    dependendency_filenames = [
        stack_files_artifacts(d)[0].filename
        for d in substack.dependencies.values()
    ]
    _symlink_frames(dependendency_filenames, run_dir)

  return first_filename, num_frames


def _setup_simple_stack(stack_id: StackHash,
                        tmp_dir: str) -> tuple[str | None, int]:
  quarantined_stacks = context.quarantined_stacks or set()
  if stack_id in quarantined_stacks:
    return None, 0

  # TODO: Get rid of this get_metadata() call.
  stack = context.stacks[stack_id]
  filenames = get_metadata(stack.frames, transform_fn=lambda i: i.filename)
  _symlink_frames(filenames, tmp_dir)

  return filenames[0], len(filenames)


# TOOD: Implement a @singledispatch version of this that distinguishes leaf
# from composite stacks.
#
# TODO: When a stack fails, keep note of it so it won't be retried for the
# exact same inputs again.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
@saves_artifacts(stack_files_artifacts)
@track_performance
def stack_files(stack_id: StackHash) -> list[Artifact]:
  assert context.output_dir

  stacks = context.stacks
  stack = stacks[stack_id]
  print(f'### Entering stack_files for {stack}')

  metadata = {
      'IMAGETYP': f'Master {stack.image_type}',
      'CONTHASH': stack.content_hash
  }

  dest_filename = output_filename(stack, context.output_dir)

  # If the stack has been quarantined, don't do anything.
  quarantined_stacks = context.quarantined_stacks or set()
  if stack_id in quarantined_stacks:
    print(f'### Refusing to stack stack {stack_id} '
          f'because it has been quarantined')
    return [FileArtifact(dest_filename, stack.content_hash)]

  os.makedirs(os.path.dirname(dest_filename), exist_ok=True)

  with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
    if type(stack) is SimpleStack:
      first_filename, num_frames = _setup_simple_stack(stack_id, tmp_dir)
    else:
      first_filename, num_frames = _setup_composite_stack(stack_id, tmp_dir)

    if num_frames < 3:
      print(f'### Cannot stack {stack_id} because it contains only '
            f'{num_frames} selected frames')
      return [FileArtifact(dest_filename, stack.content_hash)]

    if context.mock_stacking:
      file_io.conversion.convert_fits_to_xisf(first_filename,
                                              dest_filename,
                                              add_noise=True,
                                              metadata=metadata)
      print(f'### Mocked stacking, converted filename {first_filename} to '
            f'filename {dest_filename} with metadata {metadata}')

      return [FileArtifact(dest_filename, stack.content_hash)]

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
    file_io.xisf_file_io.add_xisf_metadata(filename, dest_filename, metadata)

    return [FileArtifact(dest_filename, stack.content_hash)]
