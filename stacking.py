import tempfile
import os
from pathlib import Path
import numpy as np
import ray
import errno
import os

from context import context
import pix
import position_string
import file_io
from dependency_graph import Graph

_CONTENT_HASH_KEY = 'CONTHASH'


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


# TODO: Split these functions out into a new file and unify them.
def matching_artifacts(image_hashes, fits_headers, content_hash):
  return [
      h for h, header in fits_headers.items()
      if header and header.get(_CONTENT_HASH_KEY, None) == content_hash and
      os.path.exists(image_hashes[h])
  ]


def artifact_exists(image_hashes, fits_headers, content_hash):
  return len(matching_artifacts(image_hashes, fits_headers, content_hash)) > 0


# TODO: This function hits every master once for every stack, which grows to
# be enormous. It should be cached somehow.
def find_matching_masters(stack, image_hashes, fits_headers):
  return [
      h for h, header in fits_headers.items()
      if header and header.get(_CONTENT_HASH_KEY, None) == stack.content_hash
      and os.path.exists(image_hashes[h])
  ]


def _output_filename(stack, dest_path):
  if stack.image_type == 'bias':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_bias_{stack.camera}_'
        f'{stack.mean_temperature:.0f}C_{stack.content_hash}.xisf')
  elif stack.image_type == 'dark':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_dark_{stack.camera}_'
        f'{stack.mean_temperature:.0f}C_{stack.exposure_s}s_'
        f'{stack.content_hash}.xisf')
  elif stack.image_type == 'flat':
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir,
        f'master_flat_{stack.camera}_{stack.filter}_{stack.content_hash}.xisf')
  elif stack.image_type == 'light':
    output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    return os.path.join(
        output_dir, f'master_light_{stack.target}_{ra_dec_string}_'
        f'{stack.camera}_{stack.image_scale:.02f}aspp_'
        f'{stack.rotation_angle:.2f}deg_{stack.filter}_{stack.content_hash}.xisf'
    )

  assert False, f'unknown image_type {stack.image_type}'


def _symlink_frames(filenames, dest_path):
  for filename in filenames:
    assert filename
    if not os.path.exists(filename):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'symlinking {filename} -> {dest_filename}')
    os.symlink(filename, dest_filename)


@pix.PixInsight()
def _run_wbpp(unused, *args, **kwargs):
  return pix.WeightedBatchPreprocessing(unused, *args, **kwargs)


# TODO: Include a timeout so a single bad stack (e.g. one that needs interactive
# plate-solving help) won't derail a long-running session.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def stack_files(stack_id, input_files):
  assert context.output_dir

  image_hashes = context.image_hashes
  input_files = dict(input_files)

  def _find_filename(hash):
    '''Find a file in the global image hashes, or files passed in via graph.'''
    if hash in image_hashes:
      filename = image_hashes[hash]
    elif hash in input_files:
      filename = input_files[hash]

    assert os.path.exists(filename)
    return filename

  stacks = context.stacks
  stack = stacks[stack_id]
  print(f'### Entering stack_files for {stack}, '
        f'input_files {input_files}')

  dest_filename = _output_filename(stack, context.output_dir)
  os.makedirs(os.path.dirname(dest_filename), exist_ok=True)

  metadata = {
      'IMAGETYP': f'Master {stack.image_type}',
      _CONTENT_HASH_KEY: stack.content_hash
  }

  # TODO: Put subframe selection here.

  with tempfile.TemporaryDirectory() as tmp_dir:
    first_hash = None
    if stack.frames:
      first_hash = stack.frames[0]

    hashes = stack.frames + list(stack.dependencies.values())
    print(f'### Stack frames and dependencies: {hashes}')
    filenames = [_find_filename(h) for h in hashes]
    for h, f in zip(hashes, filenames):
      print(f'hash {h} -> filename {f}')
    assert None not in filenames
    _symlink_frames(filenames, tmp_dir)

    for i, substack_id in enumerate(stack.contents):
      substack = stacks[substack_id]

      run_dir = os.path.join(tmp_dir, f'RUN_{i}')
      os.makedirs(run_dir, exist_ok=True)

      if not first_hash and substack.frames:
        first_hash = substack.frames[0]

      hashes = substack.frames + list(substack.dependencies.values())
      print(f'### Substack frames and dependencies: {hashes}')
      filenames = [_find_filename(h) for h in hashes]
      for h, f in zip(hashes, filenames):
        print(f'hash {h} -> filename {f}')

      assert None not in filenames
      _symlink_frames(filenames, run_dir)

    if context.mock_stacking:
      file_io.conversion.convert_fits_to_xisf(image_hashes[first_hash],
                                              dest_filename,
                                              add_noise=True,
                                              metadata=metadata)
      print(f'### Mocked stacking, wrote filename {dest_filename} '
            f'with metadata {metadata}')
      return stack.content_hash, dest_filename

    print(f'### Running _run_wbpp for stack {stack}')
    # TODO: Putting [] here for `filenames` is a wart.
    # pix.WeightedBatchPreprocessing([], dir=tmp_dir, outputDirectory=tmp_dir)
    _run_wbpp([], dir=tmp_dir, outputDirectory=tmp_dir)

    master_dir = os.path.join(tmp_dir, 'master')
    if not os.path.exists(master_dir):
      exception = Exception("WBPP did not produce an output master dir")
      print(f'Exception {exception}')
      key = ('stack_files', stack_id)
      context.set(key, exception)
      return None, None

    print('WBPP master dir contains:')
    for f in os.listdir(master_dir):
      print(f'  {f}')

    wbpp_masters = list(
        str(f.absolute()) for f in Path(master_dir).rglob("master*.xisf"))
    wbpp_masters.sort(key=lambda x: os.path.getmtime(x))
    filename = wbpp_masters[-1]
    print(f'last modified WBPP master {filename} -> {dest_filename}')
    file_io.xisf_file_io.add_xisf_metadata(filename, dest_filename, metadata)

    key = ('stack_files', stack_id)
    context.set(key, stack.content_hash)
    return stack.content_hash, dest_filename


# TODO: Should this function ignore the stack if it has already been stacked?
# Users may want to force re-stacking with new external software.
@ray.remote
def scatter_stack_subframes(stack_id):
  image_hashes = context.image_hashes
  fits_headers = context.fits_headers
  stacks = context.stacks
  stack = stacks[stack_id]
  print(f'scatter_stack_subframes for stack {stack}')

  def master_exists(stack_id):
    result = len(
        find_matching_masters(stacks[stack_id], image_hashes, fits_headers)) > 0
    return result

  def previously_had_exception(stack_id):
    return type(context.get(('stack_files', stack_id), None)) is Exception

  if master_exists(stack_id):
    print(f'### stack {stack} master exists')
    return {}

  if previously_had_exception(stack_id):
    print(f'### stack {stack} previously had exception')
    return {}

  # TODO: This is gross. Many different stacks have the same dependencies, but
  # they are never considered together in one big graph, where they can be
  # automatically de-duped.
  graph = {}
  graph[(stack_files,
         stack_id)] = [(stack_files, d)
                       for substack in stack.contents
                       for d in stacks[substack].dependencies.values()
                       if stacks[d].image_type != 'light' and
                       not master_exists(d) and not previously_had_exception(d)]

  return Graph(graph)
