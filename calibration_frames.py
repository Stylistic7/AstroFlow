import os
import itertools
import ray
import numpy as np
import datetime

import ray_diskcache
import runs
import clustering
import stacking
import hashing
from stack import Stack, Run, FrameHash

# 1. Cluster by lists of FITS headers:
#
#   Bias frames:
#     IMAGETYP (discrete)
#     INSTRUME (discrete),
#     CCD-TEMP (continuous, match disribution of light frames)
#
#   Dark frames:
#     IMAGETYP (discrete)
#     INSTRUME (discrete)
#     EXPOSURE (discrete)
#     CCD-TEMP (continuous, match disribution of light frames)
#
#   Flat frames:
#     IMAGETYP (discrete)
#     INSTRUME (discrete)
#
#   All calibration frames can be grouped once and then searched for frames
#   that match the lights.
#
# 2. Get a list of the runs of light frames in the stack.
#
#    These may be taken across many different dates.
#
#    For each run:
#      Get the INSTRUME, EXPOSURE from the first light frame.
#      Compute the mean and std. dev of CCD-TEMP.
#
#      Find bias, dark, and flat frames that match by the criteria above.
#
#    What's the best data structure for this?
#      I've been storing runs as start and end dates tuples, which is workable
#      but requires lookups everywhere. The user doesn't want to see
#      ['b63d76d8a25772f7', '68d2184c67161e4f', '1aea24e81cf21224'], though,
#      they want to see [1 Jan 2024 10 pm, 1 Jan 2024 11 pm]
#
#    Each run of light frames is associated with a specific run of bias frames,
#    a specific set of dark frames, and a specific set of flat frames.
#
#    Maybe calibration_frame_associations: {
#      [hashes]: {
#        'biases': [hashes],
#        'darks': [hashes],
#        'flats': [hashes],
#      }
#    }
#
# 3. When stacking calibration frames, the eventual number (like BIASRUN3) is
#    unknown, because it depends on the stack. Instead, the known information is
#    the camera, expsoure, temperature, and start datetime.
#
#    The stacked masters should be named something like:
#
#      Calibration/Master Darks/master_dark_camera_exposure_temp_start_time.xisf
#
# 4. At stacking time, each light frame should be symlinked like this:
#      light_camera_temp_exposure_target_ra_dec_image_scale_rotation_BIASRUN_0_DARKRUN_0_FLATRUN_1.fit
#
#    The master bias(es) should be symlinked like this:
#      master_bias_camera_temp_BIASRUN0.xisf
#
#    The master dark(s) should be symlinked like this:
#      master_dark_camera_exposure_temp_DARKRUN0.xisf


def _associate_bias_frames(stack, fits_headers):
  print(f'### Entering _associate_bias_frames stack {stack}')
  output_stacks = [stack]
  for run in stack.runs:
    light_frames = run.frames
    assert all(fits_headers[f]['IMAGETYP'] == 'Light' for f in light_frames)
    light_runs = runs.get_runs(light_frames, fits_headers)
    assert len(light_runs) == 1
    light_run = light_runs[0]

    bias_frames = [
        f for f, header in fits_headers.items()
        if header['IMAGETYP'] == 'Bias' and header['INSTRUME'] == stack.camera
    ]
    light_frame_temps = [(f, fits_headers[f]['CCD-TEMP']) for f in light_frames]

    bias_frames = [(f, fits_headers[f]['CCD-TEMP']) for f in bias_frames]
    bias_frames = clustering.items_in_distribution(light_frame_temps,
                                                   bias_frames,
                                                   slop=5)
    bias_frames = [f for f, _ in bias_frames]

    bias_runs = runs.get_runs(bias_frames, fits_headers)
    bias_run = bias_runs[np.argmin(
        [runs.run_time_delta(light_run, bias_run) for bias_run in bias_runs])]
    bias_frames = runs.get_frames_for_run(bias_run, bias_frames, fits_headers)

    print('### bias_frames', bias_frames)
    new_run = Run(frames=[FrameHash(h) for h in bias_frames])

    mean_temperature = np.mean(
        [fits_headers[f]['CCD-TEMP'] for f in bias_frames])

    bias_stack = Stack(runs=[new_run],
                       camera=stack.camera,
                       mean_temperature=mean_temperature,
                       width_pixels=stack.width_pixels,
                       height_pixels=stack.height_pixels)
    run.stack_dependencies.append(bias_stack.hash())
    output_stacks.append(bias_stack)

  return output_stacks


# TODO: Should bias and dark frames also be grouped into runs, and then the
# closest run selected for each run of lights?
# def _associate_bias_frames(stack, fits_headers):
#   frames_for_image_type = clustering.cluster_discrete(
#       fits_headers.keys(), key=lambda f: fits_headers[f]['IMAGETYP'])
#   frames_for_camera = clustering.cluster_discrete(
#       fits_headers.keys(), key=lambda f: fits_headers[f]['INSTRUME'])

#   # Get the camera used in the stack. Make sure the stack is for a single
#   # camera.
#   assert stack.light_frames
#   assert len(set(fits_headers[f]['INSTRUME'] for f in stack.light_frames)) == 1
#   camera = fits_headers[stack.light_frames[0]]['INSTRUME']

#   light_runs = runs.get_runs(stack.light_frames, fits_headers)
#   bias_runs = runs.get_runs(
#       set(frames_for_image_type['Bias']) & set(frames_for_camera[camera]),
#       fits_headers)
#   dark_runs = runs.get_runs(
#       set(frames_for_image_type['Dark']) & set(frames_for_camera[camera]),
#       fits_headers)

#   for light_run in light_runs:
#     light_frames = tuple(
#         sorted(
#             runs.get_frames_for_run(light_run, stack.light_frames,
#                                     fits_headers)))
#     if light_frames not in stack.calibration_frame_associations:
#       stack.calibration_frame_associations[light_frames] = {}

#     if bias_runs:
#       bias_run = bias_runs[np.argmin(
#           [runs.run_time_delta(light_run, bias_run) for bias_run in bias_runs])]
#       bias_frames = set(
#           runs.get_frames_for_run(bias_run, frames_for_image_type['Bias'],
#                                   fits_headers))
#     else:
#       bias_frames = []

#     if dark_runs:
#       dark_run = dark_runs[np.argmin(
#           [runs.run_time_delta(light_run, dark_run) for dark_run in dark_runs])]
#       dark_frames = set(
#           runs.get_frames_for_run(dark_run, frames_for_image_type['Bias'],
#                                   fits_headers))
#     else:
#       dark_frames = []

#     stack.calibration_frame_associations[light_frames][
#         'biases'] = clustering.items_in_distribution(
#             light_frames,
#             bias_frames,
#             slop=5.0,
#             key=lambda f: fits_headers[f]['CCD-TEMP'])

#   return stack

# def _associate_dark_frames(stack, fits_headers):
#   # Get the camera and exposure duration of the stack. Make sure the stack is
#   # for a single camera. Multiple exposure durations are ok.
#   #
#   # TODO: Think more about how to handle different exposure durations. Some
#   # sources say I can just toss all the exposures into WBPP with a big exposure
#   # tolerance, but I don't know how to control exposure tolerance
#   # programmatically.
#   assert stack.light_frames
#   assert len(set(fits_headers[f]['INSTRUME'] for f in stack.light_frames)) == 1
#   camera = fits_headers[stack.light_frames[0]]['INSTRUME']
#   mean_temperature = np.mean(
#       [fits_headers[f]['CCD-TEMP'] for f in stack.light_frames])
#   exposures = set(fits_headers[f]['EXPOSURE'] for f in stack.light_frames)
#   print(f'### _associate_dark_frames: stack has camera {camera} '
#         f'mean_temperature {mean_temperature} exposures {exposures}')

#   # Find all matching dark frames. Dark exposures must be equal in exposure
#   # duration to the light frames, or longer if dark scaling is used.
#   # TODO: Make dark scaling a user option.
#   dark_frames = [
#       h for h, header in fits_headers.items()
#       if {'IMAGETYP', 'INSTRUME', 'EXPOSURE', 'CCD-TEMP'} <= set(header) and
#       'dark' in header['IMAGETYP'].lower() and 'master' not in
#       header['IMAGETYP'].lower() and header['INSTRUME'] == camera and any(
#           header['EXPOSURE'] >= e for e in exposures)
#   ]
#   print(f'### Found {len(dark_frames)} dark frames for camera and exposures')

#   # Select darks with temperatures that are consistent with the distrubution
#   # of temperatures of the light frames.
#   # TODO: Maybe reject light frames that appear to be outliers in temperature.
#   dark_frames = clustering.items_in_distribution(
#       stack.light_frames,
#       dark_frames,
#       slop=5.0,
#       key=lambda f: fits_headers[f]['CCD-TEMP'])
#   print(f'### Found {len(dark_frames)} dark frames with temperatures that fit '
#         f'the distribution of the light frames')

#   stack.dark_frames = list(dark_frames)
#   if not stack.dark_frames:
#     print(f'WARNING stack has no dark frames')

#   return stack

# def _associate_flat_frames(stack, fits_headers):
#   # Get the camera and filter of the stack. Make sure the stack is for a single
#   # camera and single filter.
#   assert stack.light_frames
#   assert len(set(fits_headers[f]['INSTRUME'] for f in stack.light_frames)) == 1
#   assert len(set(fits_headers[f]['FILTER'] for f in stack.light_frames)) == 1
#   camera = fits_headers[stack.light_frames[0]]['INSTRUME']
#   filter = fits_headers[stack.light_frames[0]]['FILTER']

#   # Find all matching flat frames.
#   flat_frames = [
#       h for h, header in fits_headers.items()
#       if {'IMAGETYP', 'INSTRUME', 'FILTER'} <= set(header) and 'flat' in
#       header['IMAGETYP'].lower() and 'master' not in header['IMAGETYP'].lower()
#       and header['INSTRUME'] == camera and header['FILTER'] == filter
#   ]

#   # Find runs of light frames and flat frames. Each run of light frames will be
#   # associated with (and only one) run of flat flames. Multiple runs of light
#   # frames may be associated with the same run of flat frames.
#   imaging_runs = runs.get_runs(stack.light_frames, fits_headers)
#   flat_runs = runs.get_runs(flat_frames, fits_headers)

#   flat_frame_associations = []
#   if flat_runs:
#     for imaging_run in imaging_runs:
#       run_distances = [
#           runs.run_time_delta(imaging_run, flat_run) for flat_run in flat_runs
#       ]
#       index = np.argmin(run_distances)
#       flat_frame_associations.append((imaging_run, flat_runs[index]))

#   stack.flat_frame_associations = flat_frame_associations

#   stack.flat_frames = sorted(
#       set(
#           runs.get_frames_for_runs(
#               (flat_run for _, flat_run in flat_frame_associations),
#               flat_frames, fits_headers)))

#   return stack


@ray.remote
def associate_calibration_frames(context):
  print(f'Associating calibration frames...')

  fits_headers = ray.get(context.fits_headers)
  plate_solutions = ray.get(context.plate_solutions)
  stacks = ray.get(context.stacks)

  # At this point, stacks is just a list of stacks. The final content hashes
  # are not known yet, because the calibration frames aren't associated yet.
  output_stacks = []
  for stack in stacks:
    # TODO: There will be lots of duplicates here; does it matter?
    stacks = _associate_bias_frames(stack, fits_headers)
    print(f'### _associate_bias_frames returned {len(stacks)} stacks')
    output_stacks.extend(stacks)

    # stack = _associate_dark_frames(stack, fits_headers)
    # stack = _associate_flat_frames(stack, fits_headers)
    # stack.store_metadata(fits_headers, plate_solutions)

  stacks = {s.hash(): s for s in output_stacks}

  print(f'### associate_calibration_frames created {len(stacks)} stacks')
  for content_hash, stack in stacks.items():
    print('###', content_hash, stack)
    print(f'###   Runs: {stack.runs}')

  ray_diskcache.set('stacks', stacks)
  context.stacks = ray.put(stacks)
  return context


# TODO: Stacks of darks of different exposures can be tossed into WBPP together
# to get multiple masters. Same goes for flats with different filters, etc.
# @ray.remote
# def scatter_stack_calibration_frames(context):
#   print(f'Stacking calibration frames...')

#   fits_headers = ray.get(context.fits_headers)
#   stacks = ray.get(context.stacks)
#   print('### scatter_stack_calibration_frames: stacks', stacks)

#   results = []

# bias_masters = set(
#     tuple(sorted(stack.bias_frames)) for stack in stacks.values())
# dark_masters = set(
#     tuple(sorted(stack.dark_frames)) for stack in stacks.values())

# flat_masters = set()
# for stack in stacks.values():
#   for _, flat_run in stack.flat_frame_associations:
#     flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
#                                           fits_headers)
#     assert flat_frames
#     flat_masters.add(tuple(sorted(flat_frames)))

# # TODO: Unify this with the similar code in stacking.py.
# def artifact_exists(master):
#   content_hash = hashing.hash_list(master)
#   matching_masters = stacking.find_matching_masters(fits_headers,
#                                                     content_hash)
#   if not matching_masters:
#     return False

#   print(f'### Calibration master has existing artifacts {matching_masters}')
#   return True

# # TODO: It's hard to get the correct filename if all you have is the hashes
# # of a calibration stack. The masters should be in a better directory
# # structure to begin with, e.g. Calibration/Masters/Bias/ASI6200MM Pro/-10C/.
# results = [
#     stacking.stack_files.remote(context, master, context.output_dir)
#     for master in list(bias_masters | dark_masters | flat_masters)
#     if master and not artifact_exists(master)
# ]

# return ray.get(gather_stack_calibration_frames.remote(results))

# @ray.remote
# def gather_stack_calibration_frames(results):
#   # scatter_stack_calibration_frames() returns a list of lists. The outer list
#   # is over all of the groups that were stacked (e.g. groups of bias or dark
#   # frames). Each stacking operation could (potentially) produce more than one
#   # master file, so the result is a list of lists. The list of lists is
#   # flattened here.
#   results = ray.get(results)
#   value = list(itertools.chain.from_iterable([n for n in results]))
#   return value
