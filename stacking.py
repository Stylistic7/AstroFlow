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

  for h, f in image_hashes.items():
    print(f'### hash {h} -> filename {f}')

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


# @ray.remote(resources={'PixInsight': 1}, scheduling_strategy="SPREAD")
# def stack_stack(context, stack):
#   print(f'### Stacking subframes for {stack}')

#   image_hashes = ray.get(context.image_hashes)
#   fits_headers = ray.get(context.fits_headers)

#   with tempfile.TemporaryDirectory() as tmp_dir:
#     _symlink_stack_frames(stack, image_hashes, fits_headers, tmp_dir)

#     pix.WeightedBatchPreprocessing(dir=tmp_dir, outputDirectory=tmp_dir)

#     master_dir = os.path.join(tmp_dir, 'master')
#     if not os.path.exists(master_dir):
#       print(f'### WBPP did not produce an output master dir.')
#       return []

#     print('### WBPP master dir contains:')
#     for f in os.listdir(master_dir):
#       print(f'###   {f}')

#     filenames = list(
#         str(f) for f in Path(master_dir).rglob("*drizzle_2x_autocrop.xisf"))
#     if not filenames:
#       # TODO: Return failure, so the stack won't just be attempted again
#       # on the next run. Distinguish the cases where the stack has never been
#       # attempted from the cases where it just didn't succeed.
#       print(f'### WBPP did not produce any master file.')
#       return []

#     print('filenames', filenames)
#     if len(filenames) != 1:
#       print(f'### WARNING: stacking produced {len(filenames)} master files')

#     # TODO: If there are exposures of different durations in the stack, WBPP
#     # will produce multiple masters. Is this desirable? Should the user be
#     # given multiple masters, so they can do HDR downstream? Or should I use
#     # the WBPP exposure tolerance to make a single master?
#     output_filenames = []
#     for filename in filenames:
#       dest_filename = _output_filename(stack, context.output_dir)
#       metadata = {
#           _STACK_ID_KEY: stack.id,
#           _CONTENT_HASH_KEY: stack.hash(),
#       }
#       os.makedirs(os.path.dirname(dest_filename), exist_ok=True)
#       _add_xisf_metadata(filename, dest_filename, metadata)
#       print('### Stacking produced master', dest_filename)
#       output_filenames.append(dest_filename)

#     return output_filenames

# def construct_graph(stacks, fits_headers):
#   graph = {}
#   for stack in stacks.values():
#     graph[stack.hash()] = [
#         s for s in stack.content if s in stacks and
#         not find_matching_masters(fits_headers, hash_list(stacks[s].content))
#     ]

#   return graph


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
    ])
    # results = list(itertools.chain.from_iterable([r for r in results]))
    print('### results', results)
    context = ray.get(scatter_compute_image_metadata.remote(context, results))
    print(f'### scatter_compute_image_metadata done for {len(results)} results')

  return context

  # print('### all_results', all_results)
  # return ray.get(gather_stack_subframes.remote(all_results))

  # def artifact_exists(stack):
  #   matching_masters = find_matching_masters(fits_headers,
  #                                            stack.hash())
  #   if not matching_masters:
  #     return False

  #   print(f'### {stack} has existing artifacts {matching_masters}')

  #   assert len(matching_masters) == 1
  #   matching_master = matching_masters[0]
  #   correct_filename = _output_filename(stack, context.output_dir)
  #   if str(matching_master) != str(correct_filename):
  #     os.makedirs(os.path.dirname(correct_filename), exist_ok=True)
  #     os.rename(matching_master, correct_filename)
  #     print(f'### Renamed master {matching_master} to {correct_filename}')

  #   return True

  # TODO: Behaviors that must be correct:
  #
  # 1. The graph must automatically dedeupe runs of calibration frames that
  # are shared with many runs of light frames. I should not have to manually
  # dedupe them.
  #
  # 2. If two runs of light frames have exactly the same calibration frames,
  # they should be simply merged into one.
  #
  # 3. The topological sort must be used to produce groups of actions that can
  # be done in parallel, NOT just the static order.
  #
  # 4. Stacks that have already been stacked (i.e. the master file is located
  # on disk) should never be put into the compute graph.

  # TODO: Open questions:
  #
  # 1. Is stacking independent runs of light frames, then stacking those results,
  # the same as stacking all the runs of light frames together, with smart
  # naming matching like DARKRUN_0 etc.?
  #
  # 2. What is the best way to identify "toplevel" stacks, which no other stacks
  # depend on? Is this even worth doing? I could operate only on Targets, which
  # are composed of one or more stacks. Users won't see Stacks on the starmap,
  # just Targets and Mosaics (same as today).

  # stacks = [
  #     stack_stack.remote(context, stack)
  #     for stack in stacks.values()
  #     if not artifact_exists(stack)
  # ]
  # print(f'### Scattering {len(stacks)} stacks that need stacking...')

  # return ray.get(gather_stack_subframes.remote(stacks))


# @ray.remote
# def gather_stack_subframes(results):
#   # scatter_stack_subframes() returns a list of lists. The outer list
#   # is over all of the groups that were stacked (e.g. groups of bias or dark
#   # frames). Each stacking operation could (potentially) produce more than one
#   # master file, so the result is a list of lists. The list of lists is
#   # flattened here.
#   print('### gather_stack_subframes results', results)
#   value = list(itertools.chain.from_iterable([n for n in results]))
#   print('### gather_stack_subframes return value', value)
#   return value
