from compute_graph import build_compute_graph
import ray_diskcache
import throughput
import round_robin
import image_handling
import image_metadata
import stacking
import step_mosaic
import step_post_process_target
import step_post_process_stack
import step_sanity_check_stack
import step_flat_frame_association
import step_download_asiair
import camera_configs
import dark_and_bias_association
import autorun_log_analysis
import spatial_association
import plate_solving
import file_io
import stack_identification
from mosaics import identification
from cache import cache
from mosaic import Mosaic
from target import Target
from stack import Stack, Status
import new_dependency_graph as dg
import runs
import pix
import clustering
import position_string
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
import operator
from spherical_geometry.polygon import SphericalPolygon
import astropy.io.fits as pyfits
import traceback

import ray

import ray_diskcache
from stack_identification import identify_stacks_ray

# TODO: Prove that local changes in the Python get propagated to remote nodes
# even without pushing to git..
print('### About to call ray.init')
traceback.print_stack()
ray_context = ray.init(address='auto',
                       namespace='AstroFlow',
                       runtime_env={
                           'working_dir': '/Users/nobody/Projects/AstroFlow',
                           'excludes': [
                               '/.git/',
                               '/web_server/',
                               '/tesdata/',
                           ]
                       },
                       ignore_reinit_error=True,
                       _node_ip_address='192.168.50.3')

print('### Ray dashboard:', ray_context.dashboard_url)

global_var_actor = ray_diskcache.GlobalVarActor.options(
    name='ray_diskcache', namespace='AstroFlow', get_if_exists=True).remote()

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


# Compute graph:
#
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
        step_download_asiair.run(rig['name'], rig['ip'], search_defs, save_dir,
                                 default_filter)


def find_fits_files(input_dirs, max_frames=None):
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
            f for f in filenames if 'bad' not in Path(f.lower()).parts
        ]
        print(f'Found {len(filenames)} FITS files that are not marked bad')

        all_frames |= set(filenames)

    return list(all_frames)


# TODO: Update the plate solution if the master file has been changed,
# e.g. its image hash is different.
def find_masters(output_dir):
    print(f'Looking for master XISF files in directory {output_dir}')
    filenames = list(str(f) for f in Path(output_dir).rglob("*.xisf"))
    print(f'Found {len(filenames)} XISF files')
    filenames = [
        f for f in filenames
        if os.path.basename(f).lower().startswith('master')
    ]
    print(f'Found {len(filenames)} XISF files that are masters')

    return filenames


# TODO: Make sure the Python version has stable iteration order for dicts.
# def sort_by_observation_date(fits_headers):
#   sorted_keys = sorted(fits_headers.keys(),
#                        key=lambda h: fits_headers[h]['DATE-OBS'])
#   output = {}
#   for key in sorted_keys:
#     output[key] = fits_headers[key]

#   return output


def split_images_by_type(fits_headers):
    print('Splitting images by type...')

    def image_type(h):
        return fits_headers[h]['IMAGETYP']

    sorted_hashes = sorted(fits_headers, key=image_type)
    files = {
        k: list(g)
        for k, g in itertools.groupby(sorted_hashes, key=image_type)
    }

    light_frames = files['Light'] if 'Light' in files else []
    dark_frames = files['Dark'] if 'Dark' in files else []
    bias_frames = files['Bias'] if 'Bias' in files else []
    flat_frames = files['Flat'] if 'Flat' in files else []

    print(f'### Found {len(light_frames)} light frames')
    print(f'### Found {len(dark_frames)} dark frames')
    print(f'### Found {len(bias_frames)} bias frames')
    print(f'### Found {len(flat_frames)} flat frames')

    return light_frames, dark_frames, bias_frames, flat_frames


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


# TODO: Parallelize this.
# TODO: Move this into a new file.
# TODO: Add tests.
def cartesian_product_lights_and_flats(stacks, flat_frames, image_hashes,
                                       fits_headers, plate_solutions):
    print('### Entering cartesian_product_lights_and_flats')
    print('### stacks', stacks)

    flats_for_camera = group_by_fn(flat_frames,
                                   lambda f: fits_headers[f]['INSTRUME'])
    flats_for_filter = group_by_fn(flat_frames,
                                   lambda f: fits_headers[f]['FILTER'])

    for stack in stacks:
        key = ('trial_flat_correction', stack.id)
        if key in cache:
            continue

        print(f'### Computing light x flat Cartestian product for {stack}')
        light_runs = runs.get_runs(stack.light_frames, fits_headers)
        best_light_frames = [
            max(
                {
                    f: p
                    for f, p in plate_solutions.items() if f in runs.
                    get_frames_for_run(run, stack.light_frames, fits_headers)
                },
                key=lambda f: plate_solutions[f].num_stars)
            for run in light_runs
        ]

        if not stack.dark_frames:
            print(f'  ### Skipping {stack} because it has no dark frames')
            continue

        dark_frame = stack.dark_frames[0]
        # TODO: assert dark frame matches light frame.
        # TODO: average the darks

        # Filter flat frames with the same camera and filter as the stack.
        flats = set()
        for camera, _, _ in stack.camera_configs:
            flats |= set(flats_for_camera[camera]) & set(
                flats_for_filter[stack.filter])

        # Group the flat frames into runs.
        flat_runs = runs.get_runs(flats, fits_headers)

        # Average the flat frames in each run.
        # TODO Don't just take the first one, it may have banding.
        average_flats = [
            runs.get_frames_for_run(run, flat_frames, fits_headers)[0]
            for run in flat_runs
        ]

        # For each element in the cross product of light runs and flat runs:
        # result = {
        #   'light_run': ...',
        #   'original_image': ...,
        #   'trial_flat_corrections': [
        #     {
        #       'flat_run': ...,
        #       'corrected_image': ...,
        #     },
        #     {
        #       'flat_run': ...,
        #       'corrected_image': ...,
        #     },
        #   ]
        # }

        results = []
        # for (light_run, light_frame), (flat_run, flat_frame) in itertools.product(
        #     zip(light_runs, best_light_frames), zip(flat_runs, average_flats)):
        for light_run, light_frame in zip(light_runs, best_light_frames):
            width = 1200

            light_frame_image = image_handling.resize_array(
                image_handling._read_image(image_hashes[light_frame]), width)
            rotate = plate_solutions[light_frame].rotation_angle < 0.0
            if rotate:
                light_frame_image = np.rot90(light_frame_image, 2)

            dark_frame_image = image_handling.resize_array(
                image_handling._read_image(image_hashes[dark_frame]), width)

            light_frame_image -= dark_frame_image

            original_image = image_handling.array_to_png(
                image_handling.autostretch(light_frame_image))

            result = {
                'light_run': light_run,
                'original_image': original_image,
                'trial_flat_corrections': []
            }

            for flat_run, flat_frame in zip(flat_runs, average_flats):
                print(f'  ### Generating trial flat-correction:')
                print(f'    ### light frame {light_frame}')
                print(f'    ### dark frame {dark_frame}')
                print(f'    ### flat frame {flat_frame}')

                flat_frame_image = image_handling.resize_array(
                    image_handling._read_image(image_hashes[flat_frame]),
                    width)

                flat_frame_image -= dark_frame_image

                corrected = np.divide(light_frame_image,
                                      flat_frame_image,
                                      out=np.zeros_like(light_frame_image),
                                      where=flat_frame_image != 0)

                std = float(np.std(corrected))

                corrected = image_handling.array_to_png(
                    image_handling.autostretch(np.clip(corrected, 0.0, 1.0)))

                result['trial_flat_corrections'].append(
                    dict(flat_run=flat_run, std=std,
                         corrected_image=corrected))

            results.append(result)

        print(f'  ### Caching result for key', key)
        cache[key] = results


# TODO: Only run the automated association once. Or maybe never run it unless
# the user says to. Or maybe an empty association list just means to use the
# default, time-based method.
def associate_flat_frames(stacks, flat_frames, fits_headers):
    print('Associating flat frames...')

    flats_for_camera = group_by_fn(flat_frames,
                                   lambda f: fits_headers[f]['INSTRUME'])
    flats_for_filter = group_by_fn(flat_frames,
                                   lambda f: fits_headers[f]['FILTER'])

    for stack in stacks:
        assert stack
        flats = set()
        for camera, _, _ in stack.camera_configs:
            flats |= set(flats_for_camera[camera]) & set(
                flats_for_filter[stack.filter])

        print(f'Found {len(flats)} flat frames for camera '
              f'{camera} and filter {stack.filter}')

        step_flat_frame_association.associate_flat_frames_for_stack(
            stack, flats, fits_headers)

    return stacks


def compute_stack_metadata(stacks, fits_headers, plate_solutions):
    for stack in stacks:
        stack.store_metadata(fits_headers, plate_solutions)

    return stacks


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
#         # print(f'### cache miss on light frame {light_frame}')
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


# TOOD: Move this into another file, e.g. identify_targets.py.
def group_stacks_into_targets(stacks, plate_solutions, fits_headers):
    polygons = {
        s.id:
        SphericalPolygon.from_wcs(
            pyfits.Header(plate_solutions[s.light_frames[0]].header))
        for s in stacks.values()
    }

    # TODO: The IoU is expensive, so check cosine similarity of the center first.
    def comparison_fn(a, b):
        iou = a.intersection(b).area() / a.union(b).area()
        return iou > 0.6

    groups = spatial_association.find_groups(polygons.keys(),
                                             polygons,
                                             comparison_fn,
                                             min_component_members=1)
    print('### Found groups', groups)

    def new_target(stack_ids):
        center = stacks[stack_ids[0]].center
        print('### new_target center', center)
        name = position_string.canonical_target_string(center)
        target = Target(name=name, stack_ids=stack_ids)
        target.store_metadata(stacks)
        return target

    targets = [new_target(g) for g in groups]
    return {t.id: t for t in targets}


def group_targets_into_mosaics(targets):
    print('Identifying mosaics...')
    mosaics = identification.identify_mosiacs(targets, plot=False)
    mosaics = {m.id: m for m in mosaics}
    return mosaics


def normalize_stacks(stacks):
    for stack in stacks:
        stack.normalize()

    return stacks


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
#     print(f'### Post-processing mosaic {mosaic}!')
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
#     print(f'### Making preview for mosaic {mosaic}!')
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

#   print('### len(stacks)', len(stacks))
#   for stack in stacks:
#     print('### center is', stack.center)

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


def get_new_filenames(filenames):
    image_hashes = ray_diskcache.get('image_hashes', {})
    print(f'### Cache contains {len(image_hashes)} image hashes')

    def exists_or_none(f):
        if f and os.path.exists(f):
            return f
        else:
            return None

    prev_num_missing_files = sum(1 for f in image_hashes.values() if not f)
    image_hashes = {h: exists_or_none(f) for h, f in image_hashes.items()}
    num_missing_files = sum(1 for f in image_hashes.values() if not f)
    print(f'### {num_missing_files - prev_num_missing_files} '
          f'files have gone missing.')

    return list(set(filenames) - set(image_hashes.values()))


def main(context):
    log.info('Starting AstroFlow')

    if any(not os.path.exists(d) for d in context.input_dirs):
        print('### Input directories were not found; refusing to run.')
        return

    # print(ray.get(print_globals.remote()))

    # TODO: This is a wart.
    # TODO: Every Ray worker node will need its own cache for this.
    pix._clear_cache()

    control_modes = cache.get('control_modes', {})
    print('### Control modes are:')
    for task, mode in control_modes.items():
        print(f'### task {task} mode {mode}')

    # TODO: Find a more general way to specify the quarantine dir, or at least
    # ask the user to confirm the dir.
    quarantine_dir = os.path.join(context.input_dirs[0],
                                  'Bad') if context.quarantine else None

    # TODO: Don't just use the first input directory.
    if context.download:
        download(context.input_dirs[0], context.default_filter)

    # TODO: Maybe just call these "input files" and "output files" or something.
    all_frames = sorted(find_fits_files(context.input_dirs))

    masters = sorted(find_masters(context.output_dir))
    print(f'Found {len(masters)} masters')

    new_filenames = get_new_filenames(all_frames)
    print(f'### Importing {len(new_filenames)} files')

    # print(f'### Iterating over {len(mosaics)} mosaics')
    # for mosaic in mosaics.values():
    #   print(f'###   Entering {mosaic}')
    #   graph[step_mosaic.process_mosaic, (mosaic, targets, stacks)] = {}

    #   toplevel_tasks.append(
    #       make_mosaic_preview(image_hashes, plate_solutions, mosaic, targets,
    #                           stacks))

    #   print(f'###   Iterating over {len(mosaic.target_ids)} targets')
    #   mosaic_tasks = []
    #   for target_id in mosaic.target_ids:
    #     target = targets[target_id]
    #     print(f'###     Entering {target}')
    #     print(f'###     Iterating over {len(target.stack_ids)} stacks')
    #     target_tasks = []
    #     for stack_id in target.stack_ids:
    #       stack = stacks[stack_id]
    #       print(f'###       Entering {stack}')
    #       task_0 = stack_subframes(stack, image_hashes, fits_headers,
    #                                plate_solutions, context.output_dir)
    #       task_1 = post_process_stack(stack, context.output_dir, [task_0])
    #       target_tasks.extend([task_0, task_1])

    #     print(f'###     Created {len(target_tasks)} tasks for target')
    #     mosaic_tasks.append(
    #         post_process_target(target, stacks, context.output_dir,
    #                             target_tasks))
    #   print(f'###   Created {len(mosaic_tasks)} tasks for mosaic')
    #   toplevel_tasks.append(
    #       post_process_mosaic(image_hashes, plate_solutions, mosaic, targets,
    #                           stacks, context.output_dir, mosaic_tasks))

    # print(f'### Created {len(toplevel_tasks)} toplevel tasks')

    # TODO: This is the place to setup control modes, or to read them from the
    # user interface.
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.DISABLED
    #                   for s in stacks.values()}
    # control_modes |= {('post_process_stack', s.id): dg.ControlMode.DISABLED
    #                   for s in stacks.values()}
    # control_modes |= {('post_process_target', t.id): dg.ControlMode.DISABLED
    #                   for t in targets.values()}
    # control_modes |= {('post_process_mosaic', m.id): dg.ControlMode.DISABLED
    #                   for m in mosaics.values()}
    # control_modes |= {('make_mosaic_preview', m.id): dg.ControlMode.AUTO
    #                   for m in mosaics.values()}

    # control_modes |= {('stack_subframes', '298749827439823'): ControlMode.FORCE}
    # control_modes |= {('stack_subframes', stack.id): dg.ControlMode.DISABLED
    #                  for stack in stacks if stack.status == Status.NOT_READY}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.FORCE
    #                  for s in stacks.values() if 'IC1805' in s.target}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.FORCE
    #                  for s in stacks.values()}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.DISABLED
    #                   for s in stacks.values()}
    # control_modes |= {('stack_subframes', s.id): dg.ControlMode.FORCE
    #                   for s in stacks.values()}
    # control_modes |= {('post_process_stack', s.id): dg.ControlMode.DISABLED
    #                   for s in stacks.values() if 'IC1848' not in s.target}
    # control_modes |= {('post_process_target', t.id): dg.ControlMode.DISABLED
    #                   for t in targets.values() if 'IC1848' not in t.name}
    # control_modes |= {('post_process_stack', s.id): dg.ControlMode.DISABLED
    #                   for s in stacks.values()}
    # control_modes |= {('post_process_target', t.id): dg.ControlMode.DISABLED
    #                   for t in targets.values()}
    # control_modes |= {('post_process_mosaic', m.id): dg.ControlMode.FORCE
    #                   for m in mosaics.values() if 'IC1848' not in m.name}
    # control_modes |= {('post_process_stack', m.id): dg.ControlMode.FORCE
    #                   for m in stacks.values()}
    # control_modes |= {('post_process_mosaic', m.id): dg.ControlMode.FORCE
    #                   for m in mosaics.values() if 'IC410' in m.name}

    # print('control_modes are', control_modes)

    # def callback(results):
    #   cache['prev_results'] = results

    try:
        ray.workflow.delete(workflow_id="AstroFlow")
        print('Deleted workflow_id AstroFlow')
    except:
        pass

    print('### Starting execution')
    result = ray.workflow.run(build_compute_graph.bind(new_filenames),
                              workflow_id='AstroFlow')
    print('### Compute graph is done! Result is', result)

    # results = dg.run_graph(graph)

    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #   results = graph.run(executor, control_modes, cache.get('prev_results', {}),
    #                       callback)

    # for t, r in results.items():
    #   print(f'### results contained task {t}')

    # TODO: Provide a better API for getting the result for a given task.
    # stacks = next(
    #     r for n, r in results.items() if n == identify_stacks_ray).result
    # print('### Stacks are')
    # for s in stacks:
    #   print('###', s)

    # TODO: Close all subordinate PixInsight instances.
    print('### Compute graph is done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run astrophotography workflow')

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

    args = parser.parse_args()

    context = Context()

    # FIXME debug
    # context.max_frames = 1000

    context.input_dirs = [os.path.abspath(i) for i in args.input_dirs]
    context.output_dir = os.path.abspath(args.output_dir)
    context.quarantine = args.quarantine
    context.download = args.download
    context.default_filter = args.default_filter

    cache['context'] = context

    main(context)
