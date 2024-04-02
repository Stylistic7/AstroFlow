import tempfile
import os
from pathlib import Path
import numpy as np
from xisf import XISF
import ray
from graphlib import TopologicalSorter
import errno
import os

import pix
import position_string
from image_metadata import scatter_compute_image_metadata
import file_io
from new_dependency_graph import Graph

_STACK_ID_KEY = 'XISF:AstroFlow:StackId'
_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


# TODO: This function hits every master once for every stack, which grows to
# be enormous. It should be cached somehow.
def find_matching_masters(stack, image_hashes, fits_headers):
  return [
      h for h, header in fits_headers.items()
      if header and header.get(_CONTENT_HASH_KEY, None) == stack.content_hash
      and os.path.exists(image_hashes[h])
  ]


def _output_filename(stack, fits_headers, dest_path):
  hashes = [f for run in stack.runs for f in run.frames]

  if all(fits_headers[f]['IMAGETYP'] == 'Bias' for f in hashes):
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_bias_{stack.camera}_'
        f'{stack.mean_temperature:.0f}C_{stack.content_hash}.xisf')
  elif all(fits_headers[f]['IMAGETYP'] == 'Dark' for f in hashes):
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_dark_{stack.camera}_'
        f'{stack.mean_temperature:.0f}C_{stack.exposure_s}s_'
        f'{stack.content_hash}.xisf')
  elif all(fits_headers[f]['IMAGETYP'] == 'Flat' for f in hashes):
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir,
        f'master_flat_{stack.camera}_{stack.filter}_{stack.content_hash}.xisf')
  else:
    output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    return os.path.join(
        output_dir, f'master_light_{stack.target}_{ra_dec_string}_'
        f'{stack.camera}_{stack.image_scale:.02f}aspp_'
        f'{stack.rotation_angle:.2f}deg_{stack.filter}_{stack.content_hash}.xisf'
    )


def _symlink_frames(filenames, dest_path):
  for filename in filenames:
    assert filename
    if not os.path.exists(filename):
      raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'symlinking {filename} -> {dest_filename}')
    os.symlink(filename, dest_filename)


def _add_xisf_metadata(filename, dest_filename, metadata):
  xisf = XISF(filename)
  image = xisf.read_image(0)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()

  for k, v in metadata.items():
    file_meta[k] = {
        "id": k,
        "type": "String",
        "value": v,
    }

  XISF.write(dest_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)


# TODO: Include a timeout so a single bad stack (e.g. one that needs interactive
# plate-solving help) won't derail a long-running session.
@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def stack_files(context, stack, dest_path, metadata=None):
  print(f'Entering stack_files for {stack}')

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)

  if not metadata:
    metadata = {}

  metadata[_CONTENT_HASH_KEY] = stack.content_hash

  with tempfile.TemporaryDirectory() as tmp_dir:
    for i, run in enumerate(stack.runs):
      run_dir = os.path.join(tmp_dir, f'RUN_{i}')
      os.makedirs(run_dir, exist_ok=True)

      hashes = run.frames + run.stack_dependencies
      filenames = [image_hashes[h] for h in hashes]
      for h, f in zip(hashes, filenames):
        print(f'hash {h} -> filename {f}')

      assert None not in filenames
      _symlink_frames(filenames, run_dir)

    pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

    master_dir = os.path.join(tmp_dir, 'master')
    if not os.path.exists(master_dir):
      print(f'ERROR: WBPP did not produce an output master dir.')
      return []

    print('WBPP master dir contains:')
    for f in os.listdir(master_dir):
      print(f'  {f}')

    wbpp_masters = list(
        str(f.absolute()) for f in Path(master_dir).rglob("master*.xisf"))
    wbpp_masters.sort(key=lambda x: os.path.getmtime(x))
    filename = wbpp_masters[-1]
    dest_filename = _output_filename(stack, fits_headers, dest_path)
    print(f'last modified WBPP master {filename} -> {dest_filename}')
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    _add_xisf_metadata(filename, dest_filename, metadata)

    return dest_filename


def construct_graph(stacks):
  graph = {}
  for stack in stacks.values():
    dependencies = []
    for run in stack.runs:
      dependencies.extend(list(run.stack_dependencies))

    # TODO: It's a bit weird to have to include both kinds of hashes here.
    graph[stack.geometric_hash or stack.content_hash] = list(set(dependencies))

  return graph


@ray.remote
def scatter_stack_subframes(context):
  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)
  shopping_cart = ray.get(context.shopping_cart)
  stacks = ray.get(context.stacks)
  print(f'scatter_stack_subframes received {len(stacks)} stacks')

  # TODO: Implement in the UI.
  def is_in_shopping_cart(stack):
    return ('stack_files', stack.geometric_hash) in shopping_cart

  return context, Graph({
      gather_stack_subframes: [
          (stack_files, s)
          for s in stacks.values()
          if is_in_shopping_cart(s) and
          not find_matching_masters(s, image_hashes, fits_headers)
      ]
  })


@ray.remote
def gather_stack_subframes(context):
  return context, None
