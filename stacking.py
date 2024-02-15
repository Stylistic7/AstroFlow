import tempfile
import os
from pathlib import Path
import numpy as np
from xisf import XISF
import ray
from graphlib import TopologicalSorter

import pix
import position_string
from image_metadata import scatter_compute_image_metadata

_STACK_ID_KEY = 'XISF:AstroFlow:StackId'
_CONTENT_HASH_KEY = 'XISF:AstroFlow:StackContentHash'


def _exposure_time_string(stack):
  exposure = stack.exposure_time_s
  hours = np.floor(exposure / 3600)
  minutes = np.floor(exposure % 3600 / 60)
  return f'{hours:02.0f}h{minutes:02.0f}m'


def find_matching_masters(fits_headers, content_hash):
  return [
      h for h, header in fits_headers.items() if _CONTENT_HASH_KEY in header and
      header[_CONTENT_HASH_KEY] == content_hash
  ]


def _output_filename(stack, fits_headers, dest_path):
  hashes = [f for run in stack.runs for f in run.frames]
  start = min(fits_headers[f]['DATE-OBS'] for f in hashes)
  start = start.strftime("%Y%m%d_%H%M%S")

  if all(fits_headers[f]['IMAGETYP'] == 'Bias' for f in hashes):
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_bias_{stack.camera}_'
        f'{stack.mean_temperature:.0f}C_{start}.xisf')
  elif all(fits_headers[f]['IMAGETYP'] == 'Dark' for f in hashes):
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_dark_{stack.camera}_'
        f'{stack.mean_temperature:.0f}C_{stack.exposure_s}s_{start}.xisf')
  elif all(fits_headers[f]['IMAGETYP'] == 'Flat' for f in hashes):
    output_dir = os.path.join(dest_path, 'Stacks', 'Calibration', 'masters')
    return os.path.join(
        output_dir, f'master_flat_{stack.camera}_{stack.filter}_{start}.xisf')
  else:
    output_dir = os.path.join(dest_path, 'Stacks', stack.target, 'masters')
    ra_dec_string = position_string.ra_dec_string(*stack.center)
    return os.path.join(
        output_dir, f'master_light_{stack.target}_{ra_dec_string}_'
        f'{stack.camera}_{stack.image_scale:.02f}aspp_'
        f'{stack.rotation_angle:.2f}deg_{stack.filter}_{start}.xisf')


def _symlink_frames(filenames, dest_path):
  for filename in filenames:
    dest_filename = os.path.join(dest_path, os.path.basename(filename))
    print(f'### symlinking {filename} -> {dest_filename}')
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


@ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
def stack_files(context, stack, dest_path, metadata=None):
  print(f'### Entering stack_files for {stack}')
  print(f'### runs {stack.runs}')

  image_hashes = ray.get(context.image_hashes)
  fits_headers = ray.get(context.fits_headers)

  if not metadata:
    metadata = {}

  if _CONTENT_HASH_KEY not in metadata:
    metadata[_CONTENT_HASH_KEY] = stack.hash()

  with tempfile.TemporaryDirectory() as tmp_dir:
    for i, run in enumerate(stack.runs):
      run_dir = os.path.join(tmp_dir, f'RUN_{i}')
      os.makedirs(run_dir, exist_ok=True)

      # TODO: What about files that aren't in image_hashes?
      filenames = [
          image_hashes[f]
          for f in run.frames + run.stack_dependencies
          if f in image_hashes
      ]
      assert len(filenames) == len(
          run.frames + run.stack_dependencies), "Some images could not be found"
      _symlink_frames(filenames, run_dir)

    pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

    master_dir = os.path.join(tmp_dir, 'master')
    if not os.path.exists(master_dir):
      print(f'### WBPP did not produce an output master dir.')
      return []

    print('### WBPP master dir contains:')
    for f in os.listdir(master_dir):
      print(f'###   {f}')

    wbpp_masters = list(
        str(f.absolute()) for f in Path(master_dir).rglob("master*.xisf"))
    wbpp_masters.sort(key=lambda x: os.path.getmtime(x))
    filename = wbpp_masters[-1]
    dest_filename = _output_filename(stack, fits_headers, dest_path)
    print(f'### last modified WBPP master {filename} -> {dest_filename}')
    os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
    _add_xisf_metadata(filename, dest_filename, metadata)

    return dest_filename


def construct_graph(stacks, fits_headers):
  print('### construct_graph stacks', stacks)
  graph = {}
  for stack in stacks.values():
    dependencies = []
    for run in stack.runs:
      dependencies.extend(list(run.stack_dependencies))

    graph[stack.hash()] = list(set(dependencies))

  return graph


def group_parallel_tasks(graph):
  ts = TopologicalSorter(graph)
  ts.prepare()

  to_run = []
  while ts.is_active():
    to_run_in_parallel = []
    for node in ts.get_ready():
      to_run_in_parallel.append(node)
      ts.done(node)
    to_run.append(to_run_in_parallel)

  return to_run


@ray.remote
def scatter_stack_subframes(context):
  fits_headers = ray.get(context.fits_headers)
  stacks = ray.get(context.stacks)
  print(f'### scatter_stack_subframes received {len(stacks)} stacks')
  print('### stacks', stacks)

  parallel_tasks = group_parallel_tasks(construct_graph(stacks, fits_headers))
  print('### parallel_tasks', parallel_tasks)

  for task_group in parallel_tasks:
    print(f'### Starting {len(task_group)} stack_files tasks in parallel...')
    results = ray.get([
        stack_files.remote(context, stacks[s], context.output_dir)
        for s in task_group
        if not find_matching_masters(fits_headers, stacks[s].hash())
    ])
    print('### results', results)
    context = ray.get(scatter_compute_image_metadata.remote(context, results))
    print(f'### scatter_compute_image_metadata done for {len(results)} results')

  return context
