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

from stack import Stack, FileList
from target import Target
from mosaic import Mosaic
from cache import cache
import file_io
import plate_solving
import spatial_association
import autorun_log_analysis
import dark_and_bias_association
import camera_configs
import step_flat_frame_association
import step_sanity_check_stack
import step_mosaic
import image_handling


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


def timeit(method):
  def timed(*args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    te = time.time()
    print('%r  %2.2f s' % (method.__name__, (te - ts)))
    return result

  return timed


def memoize_by_context(fields=None):
  if fields == None:
    fields = []

  def actual_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # print(f'{func.__name__} wrapper: context is {context}')

      def hash_file_list(x):
        if x == None:
          return None
        return hashlib.sha256(bytes(str(sorted(x)), 'utf-8')).hexdigest()

      def maybe_hash(x):
        if type(x) == Context:
          return None
        elif type(x) in [FitsHeaders, PlateSolutions, ImageHashes]:
          return hash_file_list(x.keys())
        elif type(x) == FileList:
          return hash_file_list(x)
        elif type(x) == Stack:
          return ('stack', hash_file_list(x.light_frames),
                  hash_file_list(x.dark_frames), hash_file_list(x.bias_frames),
                  hash_file_list(x.flat_frames), x.flat_frame_associations)
        elif type(x) is list:
          return [maybe_hash(y) for y in x]
        elif type(x) is tuple:
          return tuple(maybe_hash(y) for y in x)
        else:
          return x

      key = (func.__name__, [maybe_hash(x) for x in args], kwargs)
      # print(f'{func.__name__} wrapper: key is {key}')

      if key not in cache or any(getattr(context, f) for f in fields):
        # print(f'### {func.__name__}: calling wrapped function')
        # print(f'{func.__name__}: key not in cache {key not in cache}')
        # print(f'{func.__name__}: context requires it '
        #       f'{any(getattr(context, f) for f in fields)}')
        result = func(*args, **kwargs)
        cache[key] = result
        return result
      else:
        # print(f'### {func.__name__}: returning cached result')
        return cache[key]

    return wrapper

  return actual_decorator


@memoize_by_context(['all_frames_changed'])
def my_func(light_frames):
  return len(light_frames)


def decorator_test():
  light_frames = ['a', 'b']
  result = my_func(light_frames)
  print('decorator_test: result', result)


def collect_results_as_dict(filenames, fn, prev_result, parallel=True):
  def run_fn(filenames):
    print(f'Running fn on {len(filenames)}')
    if parallel:
      with Pool() as p:
        results = p.map(fn, filenames)
        return {f: x for f, x in zip(filenames, results)}
    else:
      return {f: fn(f) for f in filenames}

  print('prev_result len', len(prev_result))
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


def download():
  print('Downloading files...')


@timeit
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
@timeit
def read_fits_headers(all_frames):
  print('Reading FITS headers...')
  fits_headers = collect_results_as_dict(all_frames, file_io.read_fits,
                                         cache.get('fits_headers', {}))
  cache['fits_headers'] = fits_headers
  return FitsHeaders(fits_headers)


# TODO: This will scale poorly when there are hundreds of thousands of images.
@timeit
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


@timeit
@memoize_by_context()
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


@timeit
@memoize_by_context()
def remove_frames_during_mount_movements(light_frames):
  print('Removing frames during mount movements...')

  # TODO: implement quarantining.
  settled_light_frames = (
      autorun_log_analysis.remove_frames_during_settling(light_frames))

  num_removed_files = len(light_frames) - len(settled_light_frames)
  print(f'Removed {num_removed_files} during settling periods.')

  return FileList(settled_light_frames)


# TODO: This will scale poorly when there are hundreds of thousands of images.
@timeit
def plate_solve(light_frames):
  print('Plate solving...')
  plate_solutions = collect_results_as_dict(light_frames,
                                            plate_solving.solve_field,
                                            cache.get('plate_solutions', {}))
  cache['plate_solutions'] = plate_solutions
  return PlateSolutions(plate_solutions)


def make_and_cache_thumbnail(light_frame, rotate):
  key = ('thumbnail_image', light_frame)
  cache[key] = image_handling.thumbnail(light_frame, rotate)


@timeit
def make_image_thumbnails(light_frames, plate_solutions):
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for light_frame in light_frames:
      if ('thumbnail_image', light_frame) in cache: continue
      rotate = plate_solutions[light_frame].rotation_angle < 0.0
      executor.submit(make_and_cache_thumbnail, light_frame, rotate)


@timeit
@memoize_by_context()
def identify_spatial_components(plate_solutions):
  print('Identifying spatial components...')
  spatial_components = spatial_association.find_components(plate_solutions)
  return [(k, FileList(v)) for k, v in spatial_components]


@timeit
@memoize_by_context()
def identify_stacks(spatial_components, fits_headers):
  print('Identifying stacks...')
  stacks = []
  for name, light_frames in spatial_components:
    for rig, rig_light_frames in file_io.group_by_fits_keyword(
        fits_headers, light_frames, 'RIG'):
      for filter, filter_light_frames in file_io.group_by_fits_keyword(
          fits_headers, rig_light_frames, 'FILTER'):
        print(f'spatial component {name} rig {rig} filter {filter}')
        stacks.append(
            Stack(target=name,
                  rig=rig,
                  filter=filter,
                  light_frames=FileList(filter_light_frames)))

  print(f'Identified {len(stacks)} stacks')
  return stacks


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


@timeit
@memoize_by_context()
def get_camera_configs(stacks, fits_headers):
  print('Getting camera configs...')
  for stack in stacks:
    stack.camera_configs = camera_configs.get_unique_camera_configs(
        stack.light_frames, fits_headers)

  return stacks


# TODO: Putting the for-loop body in a single function means that memoization
# will fail if even one stack changes. These steps are pretty fast, but consider
# making the loop body an independent function, and memoizing that.
@timeit
@memoize_by_context()
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
    darks = set()
    for camera, temperature, exposure in stack.camera_configs:
      darks |= set(darks_for_camera[camera]) & set(
          darks_for_temperature[temperature]) & set(
              darks_for_exposure[exposure])

    stack.dark_frames = FileList(darks)
    print(f'Found {len(stack.dark_frames)} dark frames')

  return stacks


@timeit
@memoize_by_context()
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
    biases = set()
    for camera, temperature, _ in stack.camera_configs:
      biases |= set(biases_for_camera[camera]) & set(
          biases_for_temperature[temperature])

    stack.bias_frames = FileList(biases)
    print(f'Found {len(stack.bias_frames)} bias frames')

  return stacks


@timeit
@memoize_by_context()
def associate_flat_frames(stacks, flat_frames, fits_headers):
  print('Associating flat frames...')

  flats_for_camera = group_by_fn(flat_frames,
                                 lambda f: fits_headers[f]['INSTRUME'])
  flats_for_rig = group_by_fn(flat_frames, lambda f: fits_headers[f]['RIG'])
  flats_for_filter = group_by_fn(flat_frames,
                                 lambda f: fits_headers[f]['FILTER'])

  for stack in stacks:
    flats = set()
    for camera, _, _ in stack.camera_configs:
      flats |= set(flats_for_camera[camera]) & set(
          flats_for_rig[stack.rig]) & set(flats_for_filter[stack.filter])

    stack.flat_frames = FileList(flats)
    print(f'Found {len(stack.flat_frames)} flat frames for rig {stack.rig} '
          f'and filter {stack.filter}')

    light_frame_dates = sorted(
        set(fits_headers[f]['DATE'] for f in stack.light_frames))
    flat_frame_dates = sorted(
        set(fits_headers[f]['DATE'] for f in stack.flat_frames))

    stack.flat_frame_associations = step_flat_frame_association._dates_to_strings(
        step_flat_frame_association._associate_dates(light_frame_dates,
                                                     flat_frame_dates))
    print(f'Stack {stack} has flat frame associations '
          f'{stack.flat_frame_associations}')

  return stacks


@memoize_by_context()
def compute_stack_metadata(stacks, fits_headers, plate_solutions):
  for stack in stacks:
    stack.store_metadata(fits_headers, plate_solutions)

  return stacks


def make_and_cache_stack_thumbnails(stack):
  image_key = ('stack_thumbnail_image', stack.target, stack.rig, stack.filter)
  video_key = ('stack_thumbnail_video', stack.target, stack.rig, stack.filter)

  cache[image_key], cache[
      video_key] = image_handling.stack_thumbnail_image_and_video(stack)


# TODO: Consider how changing a config setting and re-running will work.
# If I just use the (target, rig, filter) keys, they won't get re-generated.
# Figure out a good universal key strategy for all kinds of objects.
def make_stack_thumbnails_videos(stacks):
  print('Making stack thumbnails...')
  with concurrent.futures.ProcessPoolExecutor() as executor:
    for stack in stacks:
      image_key = ('stack_thumbnail_image', stack.target, stack.rig,
                   stack.filter)
      video_key = ('stack_thumbnail_video', stack.target, stack.rig,
                   stack.filter)

      if image_key in cache and video_key in cache:
        continue

      executor.submit(make_and_cache_stack_thumbnails, stack)


@memoize_by_context()
def group_stacks_into_targets(stacks):
  stacks_for_target = defaultdict(list)
  for filter in ['L', 'R', 'G', 'B', 'S', 'H', 'O']:
    for stack in [s for s in stacks if s.filter == filter]:
      stacks_for_target[stack.target].append(stack.id)

  targets = [Target(name=k, stack_ids=v) for k, v in stacks_for_target.items()]
  [t.store_metadata(stacks) for t in targets]
  return targets


# @memoize_by_context()
def group_targets_into_mosaics(targets):
  print('Identifying mosaics...')
  return step_mosaic.run(targets, plot=False)


@memoize_by_context()
def sanity_check_stacks(stacks, fits_headers):
  print('Sanity-checking stacks...')
  for stack in stacks:
    # TODO: Print failures in the correct order so the output is reason
    stack.status = step_sanity_check_stack.run(stack, fits_headers)
    if stack.status:
      print(f'Stack {stack} is good!')
    else:
      print(f'Stack {stack} failed sanity check')

  return stacks


# TODO: The master filenames on disk do not contain the hashes of the images,
# so it's impossible to know if, e.g. more flats were added since the last
# stack. Figure out some universal keying system, so we can identify stale
# results.
#
# In general, each artifact is a pure function of a bunch of other artifacts,
# e.g. light frames, flat frames, configuration, There's a dependency graph
# between artifacts and their predecessors.
#
# Right now, each step contains for loops (e.g. over stacks), but these bodies
# need to be broken apart into individual function calls.
def stack_subframes(context):
  print('Stacking subframes...')


# def post_process_stacks(context):
#   print('Post-processing stacks...')

# TODO: Implement fits_matches with fits_headers
# TODO: flat frame associations are clearly nuts


def main(context):
  decorator_test()

  download()
  all_frames = find_files(context.input_dirs)
  fits_headers = read_fits_headers(all_frames)
  image_hashes = compute_image_hashes(all_frames)
  # check_duplicates(image_hashes)
  light_frames, dark_frames, bias_frames, flat_frames = split_images_by_type(
      all_frames, fits_headers)
  light_frames = remove_frames_during_mount_movements(light_frames)
  plate_solutions = plate_solve(light_frames)
  make_image_thumbnails(light_frames, plate_solutions)
  spatial_components = identify_spatial_components(plate_solutions)
  stacks = identify_stacks(spatial_components, fits_headers)
  # measure_stacks()
  stacks = get_camera_configs(stacks, fits_headers)
  stacks = associate_dark_frames(stacks, dark_frames, fits_headers)
  stacks = associate_bias_frames(stacks, bias_frames, fits_headers)
  stacks = associate_flat_frames(stacks, flat_frames, fits_headers)
  stacks = compute_stack_metadata(stacks, fits_headers, plate_solutions)
  make_stack_thumbnails_videos(stacks)
  targets = group_stacks_into_targets(stacks)
  mosaics = group_targets_into_mosaics(targets)
  stacks = sanity_check_stacks(stacks, fits_headers)

  cache['stacks'] = stacks
  cache['targets'] = targets
  cache['mosaics'] = mosaics

  # stack_subframes()
  # post_process_stacks()


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

  main(context)