import ray
from cache import cache
import pix
import os
import argparse
from multiprocessing import Pool
import structlog
import datetime

import ray_init
from context import context
import download_asiair
import autorun_log_analysis
import compute_graph
from dependency_graph import run_graph

logger = structlog.get_logger()
log = logger

# Things that need to happen to make AstroFlow reach the "grown-up" point:
#
# - It needs to tell you what it's going to do.
# - It needs to tell you what's currently doing, and a reliable ETA.
# - It needs to tell you what it has done, so you know to look at new pics.
# - It needs to explain errors in much greater detail.
# - It needs a way to retry failed tasks.
# - Select photos and move / delete them.
# - It must not touch the user's photos, so there's no risk of bad changes.
#   - This isn't really possible; it needs to add the filter for HyperStar.
# - It needs to provide ETAs of all long-running tasks.
#
# Parts of AstroFlow that could be standalone projects:
#
# - The functional PixInsight library, pix.
# - The dependency graph and control modes stuff.
# - The mosaic merging code?
#
# Some simple metrics to pick images:
#
# - To find the best flat frame, look for the flat-corrected image with the
#   narrowest "background spike." The background must be sampled with GraXpert
#   squares. Puzzle: how can I find the best image to choose the squares?
#
# - To find the image with the least clouds, try to find the image with the
#   smallest median. This image is the "darkest." This depends upon the idea
#   clouds are always brighter than space, so not reliable.

# TODO: Give the user an option to sort mosaics / targets by date, so they
# can easily resume a previous project.

# TODO: Emit mosaic plans in Telescopius format for easy continuation.


# TODO: The adding of new files, and the removing of deleted files, should be
# broken up into multiple smaller steps. If the user moves files around, we want
# to give the user the chance to point the program at the new locations.
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
    download_asiair.run(rig['name'], rig['ip'], search_defs, save_dir,
                        default_filter)


def remove_frames_during_mount_movements(light_frames):
  print('Removing frames during mount movements...')

  # TODO: implement quarantining.
  settled_light_frames = (
      autorun_log_analysis.remove_frames_during_settling(light_frames))

  num_removed_files = len(light_frames) - len(settled_light_frames)
  print(f'Removed {num_removed_files} during settling periods.')

  return settled_light_frames


def assert_all_light_frames_in_stacks(light_frames, stacks):
  file_sets = [set(s.light_frames) for s in stacks]
  light_frames_in_stacks = set().union(*file_sets)
  # print(f'{len(light_frames)} light frames, '
  #       f'{len(light_frames_in_stacks)} light frames in stacks')
  assert set(light_frames) == light_frames_in_stacks


# TODO: Parallelize this.
# TODO: Move this into a new file.
# TODO: Add tests.
# def cartesian_product_lights_and_flats(stacks, flat_frames, image_hashes,
#                                        fits_headers, plate_solutions):
#   print('Entering cartesian_product_lights_and_flats')
#   print('stacks', stacks)

#   flats_for_camera = group_by_fn(flat_frames,
#                                  lambda f: fits_headers[f]['INSTRUME'])
#   flats_for_filter = group_by_fn(flat_frames,
#                                  lambda f: fits_headers[f]['FILTER'])

#   for stack in stacks:
#     key = ('trial_flat_correction', stack.id)
#     if key in cache:
#       continue

#     print(f'Computing light x flat Cartestian product for {stack}')
#     light_runs = runs.get_runs(stack.light_frames, fits_headers)
#     best_light_frames = [
#         max(
#             {
#                 f: p for f, p in plate_solutions.items() if f in
#                 runs.get_frames_for_run(run, stack.light_frames, fits_headers)
#             },
#             key=lambda f: plate_solutions[f].num_stars) for run in light_runs
#     ]

#     if not stack.dark_frames:
#       print(f'  Skipping {stack} because it has no dark frames')
#       continue

#     dark_frame = stack.dark_frames[0]
#     # TODO: assert dark frame matches light frame.
#     # TODO: average the darks

#     # Filter flat frames with the same camera and filter as the stack.
#     flats = set()
#     for camera, _, _ in stack.camera_configs:
#       flats |= set(flats_for_camera[camera]) & set(
#           flats_for_filter[stack.filter])

#     # Group the flat frames into runs.
#     flat_runs = runs.get_runs(flats, fits_headers)

#     # Average the flat frames in each run.
#     # TODO Don't just take the first one, it may have banding.
#     average_flats = [
#         runs.get_frames_for_run(run, flat_frames, fits_headers)[0]
#         for run in flat_runs
#     ]

#     # For each element in the cross product of light runs and flat runs:
#     # result = {
#     #   'light_run': ...',
#     #   'original_image': ...,
#     #   'trial_flat_corrections': [
#     #     {
#     #       'flat_run': ...,
#     #       'corrected_image': ...,
#     #     },
#     #     {
#     #       'flat_run': ...,
#     #       'corrected_image': ...,
#     #     },
#     #   ]
#     # }

#     results = []
#     # for (light_run, light_frame), (flat_run, flat_frame) in itertools.product(
#     #     zip(light_runs, best_light_frames), zip(flat_runs, average_flats)):
#     for light_run, light_frame in zip(light_runs, best_light_frames):
#       width = 1200

#       light_frame_image = image_handling.resize_array(
#           image_handling._read_image(image_hashes[light_frame]), width)
#       rotate = plate_solutions[light_frame].rotation_angle < 0.0
#       if rotate:
#         light_frame_image = np.rot90(light_frame_image, 2)

#       dark_frame_image = image_handling.resize_array(
#           image_handling._read_image(image_hashes[dark_frame]), width)

#       light_frame_image -= dark_frame_image

#       original_image = image_handling.array_to_png(
#           image_handling.autostretch(light_frame_image))

#       result = {
#           'light_run': light_run,
#           'original_image': original_image,
#           'trial_flat_corrections': []
#       }

#       for flat_run, flat_frame in zip(flat_runs, average_flats):
#         print(f'  Generating trial flat-correction:')
#         print(f'    light frame {light_frame}')
#         print(f'    dark frame {dark_frame}')
#         print(f'    flat frame {flat_frame}')

#         flat_frame_image = image_handling.resize_array(
#             image_handling._read_image(image_hashes[flat_frame]), width)

#         flat_frame_image -= dark_frame_image

#         corrected = np.divide(light_frame_image,
#                               flat_frame_image,
#                               out=np.zeros_like(light_frame_image),
#                               where=flat_frame_image != 0)

#         std = float(np.std(corrected))

#         corrected = image_handling.array_to_png(
#             image_handling.autostretch(np.clip(corrected, 0.0, 1.0)))

#         result['trial_flat_corrections'].append(
#             dict(flat_run=flat_run, std=std, corrected_image=corrected))

#       results.append(result)

#     print(f'  Caching result for key', key)
#     cache[key] = results

# TODO: Split these into another file.
#
# TODO: Generally speaking, all prevew-making operations should take
# precedence, before other long-running tasks like stacking. Currently, there
# is no priority system in the dependency graph, and the preview-making
# operations aren't even in the dependency graph.
#
# It would be amazing if anytime a better-quality artifact gets made, e.g.
# starting with an autostretched single image, then a stack master, that all
# downstream previews would automatically be updated, too.
#
# This could be accomplished if there is a node to get the best available
# preview, and that node emits dg.Ran.RAN when the preview is changed, or a
# new preview is available.
#
# The first layer of previews is the individual frames, autostretched. If any
# of these frames change, or are removed from a stack, the next layer of
# previews should be automatically re-generated.
#
# The second layer of previews are mosaics. Mosaics can be generated at any
# output resolution, so they could potentially be built incrementally as
# "background" tasks. AstroFlow would need to be sorta re-architected as a
# typical long-running program.
#
# The third layer of previews are the stack videos. There may be multiple
# videos per stack (e.g. one for all frames, and one just for frames selected
# for stacking, or for different choices of flat frames).
# def make_image_thumbnails(stacks,
#                           image_hashes,
#                           plate_solutions,
#                           fits_headers,
#                           width=1200):
#   with concurrent.futures.ProcessPoolExecutor() as executor:
#     for stack in stacks:
#       assert stack
#       for light_frame in stack.light_frames:
#         if ('thumbnail_image', light_frame) in cache:
#           continue
#         # print(f'cache miss on light frame {light_frame}')
#         flat_frame = image_handling._get_flat_frame_for_light_frame(
#             light_frame, stack, fits_headers)
#         dark_frame = stack.dark_frames[0] if len(stack.dark_frames) else None
#         rotate = plate_solutions[light_frame].rotation_angle < 0.0
#         executor.submit(image_handling.make_and_cache_thumbnail, image_hashes,
#                         light_frame, dark_frame, flat_frame, rotate, None,
#                         width)

# TODO: Consider how changing a config setting and re-running will work.
# If I just use the (target, rig, filter) keys, they won't get re-generated.
# Figure out a good universal key strategy for all kinds of objects.
# def make_stack_thumbnails_videos(stacks, image_hashes, plate_solutions,
#                                  fits_headers):
#   print('Making stack thumbnails...')
#   with concurrent.futures.ProcessPoolExecutor() as executor:
#     for stack in stacks.values():
#       assert stack
#       image_key = ('stack_thumbnail_image', stack.id)
#       video_key = ('stack_thumbnail_video', stack.id)

#       if image_key in cache and video_key in cache:
#         continue

#       executor.submit(image_handling.make_and_cache_stack_thumbnails, stack,
#                       image_hashes, plate_solutions)

# def sanity_check_stacks(stacks, fits_headers):
#   print('Sanity-checking stacks...')
#   for stack in stacks:
#     # TODO: Print failures in the correct order so the output is reason
#     if step_sanity_check_stack.run(stack, fits_headers):
#       print(f'Stack {stack} is good!')
#       stack.status = Status.READY
#     else:
#       print(f'Stack {stack} failed sanity check')
#       stack.status = Status.NOT_READY

#   return stacks

# graph = dg.DependencyGraph()

# def stack_subframes(stack, image_hashes, fits_headers, plate_solutions,
#                     output_dir):
#   @graph.add()
#   def inner():
#     if stack.status != Status.READY:
#       return (dg.Ran.RAN, (dg.Result.FAILURE, "Stack failed sanity checks"))

#     print(f'Stacking stack {stack}!')
#     result = stacking.run(stack, image_hashes, fits_headers, plate_solutions,
#                           output_dir)
#     print(f'stacking.run returned {result}')
#     return result

#   inner.id = ('stack_subframes', stack.id)
#   return inner

# def post_process_stack(stack, output_dir, upstream_tasks):
#   @graph.add(*upstream_tasks)
#   def inner():
#     print(f'Post-processing stack {stack}!')
#     return step_post_process_stack.post_process_stack(stack, output_dir)

#   inner.id = ('post_process_stack', stack.id)
#   return inner

# def post_process_target(target, stacks, output_dir, upstream_tasks):
#   @graph.add(*upstream_tasks)
#   def inner():
#     print(f'Post-processing target {target}!')
#     return step_post_process_target.run(target, stacks, output_dir)

#   inner.id = ('post_process_target', target.id)
#   return inner

# def post_process_mosaic(image_hashes, plate_solutions, mosaic, targets, stacks,
#                         output_dir, upstream_tasks):
#   @graph.add(*upstream_tasks)
#   def inner():
#     print(f'Post-processing mosaic {mosaic}!')
#     return step_mosaic.process_mosaic(image_hashes,
#                                       plate_solutions,
#                                       mosaic,
#                                       targets,
#                                       stacks,
#                                       preview=False,
#                                       output_dir=output_dir)

#   inner.id = ('post_process_mosaic', mosaic.id)
#   return inner

# def make_mosaic_preview(image_hashes, plate_solutions, mosaic, targets,
#                         stacks):
#   @graph.add()
#   def inner():
#     print(f'Making preview for mosaic {mosaic}!')
#     return step_mosaic.process_mosaic(image_hashes,
#                                       plate_solutions,
#                                       mosaic,
#                                       targets,
#                                       stacks,
#                                       preview=True)

#   inner.id = ('make_mosaic_preview', mosaic.id)
#   return inner

# def group_images(image_hashes,
#                  plate_solutions,
#                  fits_headers,
#                  light_frames,
#                  dark_frames,
#                  bias_frames,
#                  flat_frames,
#                  quarantine_dir=None):
#   # TODO: Make this more granular.
#   # light_frames = remove_frames_during_mount_movements(light_frames)

#   plate_solutions = remove_bad_images(image_hashes, plate_solutions)

#   # Remove light frames that don't have good plate solutions.
#   light_frames = list(set(light_frames).intersection(plate_solutions.keys()))

#   spatial_components = spatial_association.find_components_by_cosine_similarity(
#       light_frames, plate_solutions, fits_headers)

#   stacks = stack_identification.identify_stacks(spatial_components,
#                                                 plate_solutions, fits_headers)

#   stacks = get_camera_configs(stacks, fits_headers)
#   stacks = associate_dark_frames(stacks, dark_frames, fits_headers)
#   stacks = associate_bias_frames(stacks, bias_frames, fits_headers)
#   stacks = associate_flat_frames(stacks, flat_frames, fits_headers)
#   stacks = compute_stack_metadata(stacks, fits_headers, plate_solutions)
#   stacks = normalize_stacks(stacks)

#   print('len(stacks)', len(stacks))
#   for stack in stacks:
#     print('center is', stack.center)

#   # TODO: Put these in the dependency graph.
#   make_image_thumbnails(stacks, image_hashes, plate_solutions, fits_headers)

#   stacks = sanity_check_stacks(stacks, fits_headers)

#   # TODO: Eliminate this shape change.
#   stacks = {stack.id: stack for stack in stacks}

#   # TODO: Put these in the dependency graph.
#   make_stack_thumbnails_videos(stacks, image_hashes, plate_solutions,
#                                fits_headers)

#   targets = group_stacks_into_targets(stacks, plate_solutions, fits_headers)
#   for target in targets:
#     print(f'Target {target}')

#   mosaics = group_targets_into_mosaics(targets)

#   return stacks, targets, mosaics


def populate_context():
  # TODO: Put these knobs in the UI.
  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160
  context.subframe_mosaic_samples = 1000

  # TODO: Should be dynamic based on the viewbox size.
  context.master_mosaic_samples = 10000

  # TODO: Don't just use the first input directory.
  if context.download:
    download(context.input_dirs[0], context.default_filter)

  if context.stacks == None:
    context.regroup = True

  if not context.contains('quarantined_files'):
    context.quarantined_files = set()

  # TODO: Add more comprehensive validation.
  # assert not any(type(p) is str for p in context.plate_solutions.values())
  # assert None not in context.image_hashes.values()

  print(f'Cache contains {len(context.quarantined_files or set())} '
        f'quarantined files')
  print(f'Cache contains {len(context.image_hashes or {})} image hashes')
  print(f'Cache contains {len(context.fits_headers or {})} FITS headers')
  print(f'Cache contains {len(context.plate_solutions or {})} plate solutions')
  print(
      f'Cache contains {len(context.shopping_cart or [])} items in shopping cart'
  )
  print(f'Cache contains {len(context.spatial_groups or [])} spatial groups')
  print(f'Cache contains {len(context.stacks or {})} stacks')
  print(f'Cache contains {len(context.targets or {})} targets')
  print(f'Cache contains {len(context.mosaics or {})} mosaics')


def main():
  log.info('Starting AstroFlow')

  # TODO: This is a wart.
  # TODO: Every Ray worker node will need its own cache for this.
  # TODO: This doesn't start PixInsight on remote workers.
  pix.start_pixinsight_instances()

  populate_context()

  print('### Starting compute graph execution')
  graph = compute_graph.build_compute_graph()
  run_graph(graph)
  print('### Compute graph is done! context is', context)

  # TODO: Close all subordinate PixInsight instances.


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
      help=('True to quarantine images that cannot be plate solved, or are '
            'duplicates.'))
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
  parser.add_argument('-r',
                      '--regroup',
                      dest='regroup',
                      action='store_true',
                      help='True to regroup images.')

  args = parser.parse_args()

  # This is a hack that waits until the ContextActor is actually alive.
  cache_keys = context.get_cache_keys()

  context.input_dirs = [os.path.abspath(i) for i in args.input_dirs]
  context.output_dir = os.path.abspath(args.output_dir)
  context.quarantine = args.quarantine
  context.download = args.download
  context.regroup = args.regroup
  context.default_filter = args.default_filter

  main()
