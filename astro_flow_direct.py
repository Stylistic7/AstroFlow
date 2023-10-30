import os
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import itertools
from multiprocessing import Pool
import functools
import hashlib
from collections import defaultdict
import concurrent.futures
import datetime
import pytz
import numpy as np
import structlog

import pix
import dependency_graph as dg
from stack import Stack, Status, FileList
from target import Target
from mosaic import Mosaic
from cache import cache
import file_io
import plate_solving
import spatial_association
import autorun_log_analysis
import dark_and_bias_association
import camera_configs
import step_download_asiair
import step_flat_frame_association
import step_sanity_check_stack
import step_post_process_stack
import step_post_process_target
import step_mosaic
import step_stack
import image_handling

logger = structlog.get_logger()
log = logger


# TODO: Building up a context object that is "parallel" to the disk cache
# is an anti-pattern. The only things that might change are:
#
# The input files
# The user config
#
# Some steps are conditioned on one, or the other, or both.
#
# There should be a way to just skip over
@dataclass
class Context:
  input_dirs: list[str] = field(default_factory=list)
  output_dir: str = None

  default_filter: str = None

  # TODO: Move to config.
  max_frames: int = None

  all_frames_changed: bool = False
  config_changed: bool = False


class FitsHeaders(dict):
  pass


class PlateSolutions(dict):
  pass


class ImageHashes(dict):
  pass


def collect_results_as_dict(filenames, fn, prev_result, parallel=True):
  log = logger.bind(fn=fn.__name__)

  def run_fn(filenames):
    log.info(f'running fn', num_files=len(filenames))
    if parallel:
      with Pool() as p:
        results = p.map(fn, filenames)
        return {f: x for f, x in zip(filenames, results)}
    else:
      return {f: fn(f) for f in filenames if fn(f)}

  log.info(f'prev_result', num_files=len(prev_result))
  result = {k: v for k, v in prev_result.items() if os.path.exists(k)}
  num_removed_files = len(prev_result) - len(result)
  if num_removed_files:
    print(f'Removed {num_removed_files} that no longer exist.')

  added_files = set(filenames) - set(prev_result.keys())
  if len(added_files):
    print(f'Computing fn for {len(added_files)} new images...')
    result |= run_fn(added_files)
  else:
    print('no work to do')

  print(f'collect_results_as_dict done, len', len(result))
  return result


def download(save_dir, default_filter):
  print('Downloading files...')
  # TODO: Put these into the config file, filled in by the web UI.
  #
  # TODO: Allow the user to pick the rig when they begin an import from an
  # ASIAIR.
  #
  # TODO: Eliminate the RIG keyword entirely. Instead, only associate images
  # into stacks when they ~the same image scale and resolution. Allow users
  # to give each combination a mnemonic name.
  #
  # TODO: Automatically apply Drizzle for under-sampled images.
  #
  # TODO: Scan network for ASIAIR devices, so the user doesn't have to set
  # static IPs.
  rigs = [
      {
          'name': 'RC71',
          'ip': '192.168.50.15',
      },
      {
          'name': 'C11',
          'ip': '192.168.50.16',
      },
  ]

  search_defs = [{
      'volume': 'EMMC%20Images',
      'log_dir': 'log',
      'fits_paths': ['Autorun', 'Plan']
  }, {
      'volume': 'TF%20Images',
      'log_dir': 'ASIAIR/log',
      'fits_paths': ['ASIAIR/Autorun', 'ASIAIR/Plan']
  }]

  # TODO: Mount the volumes, then repeatedly copy files from them, e.g. every
  # half hour. Any imaging runs that are closed (e.g. a pane has been finished)
  # are then ready to stack. Stacking and downloading must be done in parallel,
  # though.
  for rig in rigs:
    step_download_asiair.run(rig['name'], rig['ip'], search_defs, save_dir,
                             default_filter)


def find_files(input_dirs, max_frames=None):
  print('Finding files...')
  all_frames = set()
  for input_dir in input_dirs:
    print(f'Looking for FITS files in directory {input_dir}')

    filenames = list(str(f) for f in Path(input_dir).rglob("*.fit"))
    print(f'Found {len(filenames)} FITS files')
    filenames = [
        f for f in filenames
        if not os.path.basename(f).lower().startswith('master')
    ]
    print(f'Found {len(filenames)} FITS files that are not masters')
    filenames = [
        f for f in filenames
        if os.path.split(os.path.dirname(f))[1].lower() != 'bad'
    ]
    print(f'Found {len(filenames)} FITS files that are not marked bad')

    all_frames |= set(filenames)

  return FileList(all_frames)


# TODO: This will scale poorly when there are hundreds of thousands of images.
def read_fits_headers(all_frames):
  structlog.contextvars.bind_contextvars(task='read_fits_headers')
  fits_headers = collect_results_as_dict(all_frames, file_io.read_fits,
                                         cache.get('fits_headers', {}))
  cache['fits_headers'] = fits_headers
  return FitsHeaders(fits_headers)


def sort_by_observation_date(all_frames, fits_headers):
  return sorted(all_frames, key=lambda f: fits_headers[f]['DATE-OBS'])


# TODO: This will scale poorly when there are hundreds of thousands of images.
def compute_image_hashes(all_frames):
  print('Computing image hashes...')
  image_hashes = collect_results_as_dict(all_frames,
                                         file_io.hash_of_image_data,
                                         cache.get('image_hashes', {}))
  cache['image_hashes'] = image_hashes
  return ImageHashes(image_hashes)


def check_duplicates(image_hashes):
  print('Checking for duplicate images...')
  # TODO: Implement


def split_images_by_type(all_frames, fits_headers):
  print('Splitting images by type...')

  def image_type(f):
    return fits_headers[f]['IMAGETYP']

  filenames = sorted(all_frames, key=image_type)
  files = {k: list(g) for k, g in itertools.groupby(filenames, key=image_type)}

  light_frames = FileList(files['Light']) if 'Light' in files else []
  dark_frames = FileList(files['Dark']) if 'Dark' in files else []
  bias_frames = FileList(files['Bias']) if 'Bias' in files else []
  flat_frames = FileList(files['Flat']) if 'Flat' in files else []

  return light_frames, dark_frames, bias_frames, flat_frames


def remove_frames_during_mount_movements(light_frames):
  print('Removing frames during mount movements...')

  # TODO: implement quarantining.
  settled_light_frames = (
      autorun_log_analysis.remove_frames_during_settling(light_frames))

  num_removed_files = len(light_frames) - len(settled_light_frames)
  print(f'Removed {num_removed_files} during settling periods.')

  return FileList(settled_light_frames)


# TODO: This will scale poorly when there are hundreds of thousands of images.
def plate_solve(light_frames):
  print('Plate solving...')
  plate_solutions = collect_results_as_dict(light_frames,
                                            plate_solving.solve_field,
                                            cache.get('plate_solutions', {}))

  # TODO: Implement quarantining.
  # num_files = len(plate_solutions)
  # plate_solutions = {f: p for f, p in plate_solutions.items() if p}
  # num_good_files = len(plate_solutions)
  # print(f'### There were {num_files} plate solutions before removing None, '
  #       f'and {num_good_files} after')

  cache['plate_solutions'] = plate_solutions
  return PlateSolutions(plate_solutions)


def remove_unsolvable_light_frames(light_frames, plate_solutions):
  solved = [f for f in light_frames if plate_solutions[f] is not None]
  # print(f'### {len(light_frames)}, {len(solved)} plate-solved light frames')
  return solved


def identify_spatial_components(plate_solutions, fits_headers):
  print('Identifying spatial components...')
  spatial_components = spatial_association.find_components(
      plate_solutions, fits_headers)
  return [(k, FileList(v)) for k, v in spatial_components.items()]


def identify_stacks(spatial_components, fits_headers):
  print('Identifying stacks...')
  stacks = []
  for name, light_frames in spatial_components:
    for rig, rig_light_frames in file_io.group_by_fits_keyword(
        fits_headers, light_frames, 'RIG'):
      for filter, filter_light_frames in file_io.group_by_fits_keyword(
          fits_headers, rig_light_frames, 'FILTER'):
        stacks.append(
            Stack(target=name,
                  rig=rig,
                  filter=filter,
                  light_frames=FileList(filter_light_frames)))

  print(f'Identified {len(stacks)} stacks')
  return stacks


def assert_all_light_frames_in_stacks(light_frames, stacks):
  file_sets = [set(s.light_frames) for s in stacks]
  light_frames_in_stacks = set().union(*file_sets)
  # print(f'### {len(light_frames)} light frames, '
  #       f'{len(light_frames_in_stacks)} light frames in stacks')
  assert set(light_frames) == light_frames_in_stacks


def process_in_parallel(fn, elements, key_func=lambda x: x):
  # Find all of the elements that are already in the cache.
  prev_results = []
  to_process = []
  for e in elements:
    key = (fn.__name__, key_func(e))
    if key in cache:
      prev_results.append(cache[key])
    else:
      to_process.append(e)

  print(f'process_in_parallel: {len(prev_results)} prev results, '
        f'{len(to_process)} to process')

  with Pool() as p:
    new_results = p.map(fn, to_process)
    for e, r in zip(to_process, new_results):
      key = (fn.__name__, key_func(e))
      cache[key] = r
    return prev_results + new_results


def measure_stacks(context):
  print('Measuring images in stacks...')


def group_by_fn(filenames, fn):
  filenames = sorted(filenames, key=fn)
  return defaultdict(
      list, {k: list(g)
             for k, g in itertools.groupby(filenames, key=fn)})


def get_camera_configs(stacks, fits_headers):
  print('Getting camera configs...')
  for stack in stacks:
    assert stack
    stack.camera_configs = camera_configs.get_unique_camera_configs(
        stack.light_frames, fits_headers)

  return stacks


def associate_dark_frames(stacks, dark_frames, fits_headers):
  print(f'Associating dark frames...')

  def rounded_temperature(f):
    temperature = fits_headers[f]['CCD-TEMP']
    multiple = 10.0
    return int(multiple * round(temperature / multiple))

  darks_for_camera = group_by_fn(dark_frames,
                                 lambda f: fits_headers[f]['INSTRUME'])
  darks_for_temperature = group_by_fn(dark_frames, rounded_temperature)
  darks_for_exposure = group_by_fn(dark_frames,
                                   lambda f: fits_headers[f]['EXPOSURE'])

  for stack in stacks:
    assert stack
    darks = set()
    for camera, temperature, exposure in stack.camera_configs:
      darks |= set(darks_for_camera[camera]) & set(
          darks_for_temperature[temperature]) & set(
              darks_for_exposure[exposure])

    stack.dark_frames = FileList(darks)
    print(f'Found {len(stack.dark_frames)} dark frames')

  return stacks


def associate_bias_frames(stacks, bias_frames, fits_headers):
  print(f'Associating bias frames...')

  def rounded_temperature(f):
    temperature = fits_headers[f]['CCD-TEMP']
    multiple = 10.0
    return int(multiple * round(temperature / multiple))

  biases_for_camera = group_by_fn(bias_frames,
                                  lambda f: fits_headers[f]['INSTRUME'])
  biases_for_temperature = group_by_fn(bias_frames, rounded_temperature)

  for stack in stacks:
    assert stack
    biases = set()
    for camera, temperature, _ in stack.camera_configs:
      biases |= set(biases_for_camera[camera]) & set(
          biases_for_temperature[temperature])

    stack.bias_frames = FileList(biases)
    print(f'Found {len(stack.bias_frames)} bias frames')

  return stacks


def associate_flat_frames(stacks, flat_frames, fits_headers):
  print('Associating flat frames...')

  flats_for_camera = group_by_fn(flat_frames,
                                 lambda f: fits_headers[f]['INSTRUME'])
  flats_for_rig = group_by_fn(flat_frames, lambda f: fits_headers[f]['RIG'])
  flats_for_filter = group_by_fn(flat_frames,
                                 lambda f: fits_headers[f]['FILTER'])

  for stack in stacks:
    assert stack
    flats = set()
    for camera, _, _ in stack.camera_configs:
      flats |= set(flats_for_camera[camera]) & set(
          flats_for_rig[stack.rig]) & set(flats_for_filter[stack.filter])

    print(f'Found {len(flats)} flat frames for rig '
          f'{stack.rig} and filter {stack.filter}')

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flats, fits_headers)

  return stacks


def compute_stack_metadata(stacks, fits_headers, plate_solutions):
  for stack in stacks:
    stack.store_metadata(fits_headers, plate_solutions)

  return stacks


def make_image_thumbnails(stacks, plate_solutions, fits_headers, width=640):
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for stack in stacks:
      assert stack
      for light_frame in stack.light_frames:
        if ('thumbnail_image', light_frame) in cache: continue
        # print(f'### cache miss on light frame {light_frame}')
        flat_frame = image_handling._get_flat_frame_for_light_frame(
            light_frame, stack, fits_headers)
        dark_frame = stack.dark_frames[0] if len(stack.dark_frames) else None
        rotate = plate_solutions[light_frame].rotation_angle < 0.0
        executor.submit(image_handling.make_and_cache_thumbnail, light_frame,
                        dark_frame, flat_frame, rotate, None, width)


# TODO: Consider how changing a config setting and re-running will work.
# If I just use the (target, rig, filter) keys, they won't get re-generated.
# Figure out a good universal key strategy for all kinds of objects.
def make_stack_thumbnails_videos(stacks, plate_solutions, fits_headers):
  print('Making stack thumbnails...')
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for stack in stacks.values():
      assert stack
      image_key = ('stack_thumbnail_image', stack.id)
      video_key = ('stack_thumbnail_video', stack.id)

      if image_key in cache and video_key in cache:
        continue

      executor.submit(image_handling.make_and_cache_stack_thumbnails, stack)


def group_stacks_into_targets(stacks):
  stacks_for_target = defaultdict(list)
  for filter in ['L', 'R', 'G', 'B', 'S', 'H', 'O']:
    for stack in [s for s in stacks.values() if s.filter == filter]:
      stacks_for_target[stack.target].append(stack.id)

  targets = [Target(name=k, stack_ids=v) for k, v in stacks_for_target.items()]
  [t.store_metadata(stacks) for t in targets]
  return {t.id: t for t in targets}


def group_targets_into_mosaics(targets):
  print('Identifying mosaics...')
  result, result_dict = step_mosaic.run(targets, plot=False)

  if result != dg.Result.SUCCESS:
    print('step_mosaic.run() failed')
    return []

  mosaics = {m.id: m for m in result_dict['mosaics']}
  return mosaics


def sanity_check_stacks(stacks, fits_headers):
  print('Sanity-checking stacks...')
  for stack in stacks:
    # TODO: Print failures in the correct order so the output is reason
    if step_sanity_check_stack.run(stack, fits_headers):
      print(f'Stack {stack} is good!')
      stack.status = Status.READY
    else:
      print(f'Stack {stack} failed sanity check')
      stack.status = Status.NOT_READY

  return stacks


graph = dg.DependencyGraph()


def stack_subframes(stack, fits_headers, output_dir):
  @graph.add()
  def inner():
    if stack.status != Status.READY:
      return (dg.Ran.RAN, (dg.Result.FAILURE, "Stack failed sanity checks"))

    # TODO: It might be nicer to just assume DISABLED on these tasks if their
    # output files exist. Then the prev_results just pass through?
    # if not step_stack.needs_to_run(stack, output_dir):
    #   _, prev_result = prev_results.get_prev_result('stack_subframes',
    #                                                 stack.id)
    #   return (dg.Ran.DID_NOT_RUN, prev_result)

    print(f'Stacking stack {stack}!')
    result = step_stack.run(stack, fits_headers, output_dir)
    print(f'step_stack.run returned {result}')
    return result

  inner.id = ('stack_subframes', stack.id)
  return inner


def post_process_stack(stack, upstream_tasks):
  @graph.add(*upstream_tasks)
  def inner():
    # if not step_post_process_stack.needs_to_run(stack, output_dir):
    #   _, prev_result = prev_results.get_prev_result('post_process_stack',
    #                                                 stack.id)
    #   return (dg.Ran.DID_NOT_RUN, prev_result)

    print(f'Post-processing stack {stack}!')
    return step_post_process_stack.post_process_stack(stack)

  inner.id = ('post_process_stack', stack.id)
  return inner


def post_process_target(target, stacks, output_dir, upstream_tasks):
  @graph.add(*upstream_tasks)
  def inner():
    print(f'Post-processing target {target}!')
    return step_post_process_target.run(target, stacks, output_dir)

  inner.id = ('post_process_target', target.id)
  return inner


def post_process_mosaic(mosaic, targets, stacks, output_dir, upstream_tasks):
  @graph.add(*upstream_tasks)
  def inner():
    print(f'### Post-processing mosaic {mosaic}!')
    return step_mosaic.process_mosaic(mosaic, targets, stacks, output_dir)

  inner.id = ('post_process_mosaic', mosaic.id)
  return inner


def main(context, default_control_mode=dg.ControlMode.AUTO):
  log.info('Starting AstroFlow')

  # TODO: This is a wart.
  pix._clear_cache()

  # TODO: Don't just use the first input directory.
  if context.download:
    download(context.input_dirs[0], context.default_filter)

  all_frames = find_files(context.input_dirs)
  fits_headers = read_fits_headers(all_frames)

  all_frames = sort_by_observation_date(all_frames, fits_headers)

  image_hashes = compute_image_hashes(all_frames)
  # check_duplicates(image_hashes)

  light_frames, dark_frames, bias_frames, flat_frames = split_images_by_type(
      all_frames, fits_headers)

  # TODO: Make this more granular.
  # light_frames = remove_frames_during_mount_movements(light_frames)

  plate_solutions = plate_solve(light_frames)
  light_frames = remove_unsolvable_light_frames(light_frames, plate_solutions)

  spatial_components = identify_spatial_components(plate_solutions,
                                                   fits_headers)
  # assert_all_light_frames_in_spatial_components(light_frames,
  #                                               spatial_components)

  stacks = identify_stacks(spatial_components, fits_headers)
  # assert_all_light_frames_in_stacks(light_frames, stacks)

  # measure_stacks()

  stacks = get_camera_configs(stacks, fits_headers)
  stacks = associate_dark_frames(stacks, dark_frames, fits_headers)
  stacks = associate_bias_frames(stacks, bias_frames, fits_headers)
  stacks = associate_flat_frames(stacks, flat_frames, fits_headers)
  stacks = compute_stack_metadata(stacks, fits_headers, plate_solutions)

  make_image_thumbnails(stacks, plate_solutions, fits_headers)

  stacks = sanity_check_stacks(stacks, fits_headers)

  # At this point, the stacks should be fingerprinted. If a stack with the same
  # fingerprint is stored in the cache, we'll load it and use it. Otherwise,
  # the stack will be considered new.
  stacks = {stack.id: stack for stack in stacks}
  # for stack in stacks.values():
  #   if stack.id in cache.get('stacks', {}):
  #     stacks[stack.id] = cache['stacks'][stack.id]
  #     print(f'### Loaded stack {stack} from cache')

  make_stack_thumbnails_videos(stacks, plate_solutions, fits_headers)

  # stacks = load_stacks_context(stacks)

  targets = group_stacks_into_targets(stacks)
  mosaics = group_targets_into_mosaics(targets)

  graph.default_control_mode = default_control_mode

  # TODO: Consider how to make this multi-threaded.
  toplevel_tasks = []
  for mosaic in mosaics.values():
    mosaic_tasks = []
    for target_id in mosaic.target_ids:
      target = targets[target_id]
      target_tasks = []
      for stack_id in target.stack_ids:
        stack = stacks[stack_id]
        task = stack_subframes(stack, fits_headers, context.output_dir)
        target_tasks.append(post_process_stack(stack, [task]))
      mosaic_tasks.append(
          post_process_target(target, stacks, context.output_dir,
                              target_tasks))
    toplevel_tasks.append(
        post_process_mosaic(mosaic, targets, stacks, context.output_dir,
                            mosaic_tasks))
    print(f'Created {len(toplevel_tasks)} toplevel tasks')

    control_modes = {}

    for stack in stacks.values():
      stack.needs_to_run = step_stack.needs_to_run(stack, context.output_dir)

    for target in targets.values():
      target.needs_to_run = any(
          stacks[s].needs_to_run for s in target.stack_ids)

    for mosaic in mosaics.values():
      if any(targets[t].needs_to_run for t in mosaic.target_ids):
        mosaic.status = Status.READY
      else:
        mosaic.status = Status.NOT_READY

    # These are for the web server to read.
    cache['stacks'] = stacks
    cache['targets'] = targets
    cache['mosaics'] = mosaics

    # TODO: This is the place to setup control modes, or to read them from the
    # user interface.
    # control_modes |= {('stack_subframes', '298749827439823'): ControlMode.FORCE}
    # control_modes |= {('stack_subframes', stack.id): dg.ControlMode.DISABLED
    #                  for stack in stacks if stack.status == Status.NOT_READY}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.FORCE
    #                  for s in stacks.values() if 'IC1805' in s.target}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.FORCE
    #                  for s in stacks.values()}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.DISABLED
    #                   for s in stacks.values()}
    control_modes |= {('stack_subframes', s.id): dg.ControlMode.DISABLED
                      for s in stacks.values() if 'IC1848' not in s.target}
    # control_modes |= {('post_process_stack', s.id): dg.ControlMode.FORCE
    #                   for s in stacks.values() if 'IC1848' in s.target}
    # control_modes |= {('post_process_target', t.id): dg.ControlMode.FORCE
    #                   for t in targets.values() if 'IC1848' in t.name}
    # control_modes |= {('post_process_mosaic', m.id): dg.ControlMode.FORCE
    #                   for m in mosaics.values() if 'IC1848' in m.name}
    print('control_modes are', control_modes)

    print('Starting execution')
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      results = graph.run(executor, control_modes)

    # TODO: These need to be stored after each operation, not just after
    # the entire graph is done. Maybe I can add a decorator to the inner
    # functions or something?
    cache['stacks'] = stacks
    cache['targets'] = targets
    cache['mosaics'] = mosaics


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run astrophotography workflow')

  parser.add_argument(
      '-i',
      '--input-dir',
      required=True,
      dest='input_dirs',
      action='append',
      help='Directories containing raw FITS images and log files.')

  parser.add_argument('-o',
                      '--output-dir',
                      required=True,
                      dest='output_dir',
                      action='store',
                      help='Output directory')
  parser.add_argument(
      '-q',
      '--quarantine',
      dest='quarantine',
      action='store_true',
      help=('True to quarantine images that cannot be plate solved, or were '
            'taken during settling periods.'))
  parser.add_argument('-d',
                      '--download',
                      dest='download',
                      action='store_true',
                      help='True to download images from ASIAIRs.')
  parser.add_argument('-f',
                      '--default-filter',
                      dest='default_filter',
                      action='store',
                      help='Default filter')

  args = parser.parse_args()

  context = Context()

  # FIXME debug
  # context.max_frames = 1000

  context.input_dirs = [os.path.abspath(i) for i in args.input_dirs]
  context.output_dir = os.path.abspath(args.output_dir)
  context.download = args.download
  context.default_filter = args.default_filter

  cache['context'] = context

  main(context)