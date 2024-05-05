import ray
import numpy as np
from copy import copy

import file_io
import ray_diskcache
import runs
import clustering
from stack import Stack, Run, FrameHash
import stack_identification


def _associate(context,
               stack,
               image_type,
               fits_headers,
               stack_dependencies=None,
               debug=False):
  if debug:
    print(f'Entering _associate for {stack} with {len(stack.runs)} runs')
    for run in stack.runs:
      print(f'run {run}')

  if not stack_dependencies:
    stack_dependencies = []

  assert None not in stack_dependencies

  if image_type == 'bias':
    match_temperature, match_exposure, match_filter = True, False, False
  elif image_type == 'dark':
    match_temperature, match_exposure, match_filter = True, True, False
  elif image_type == 'flat':
    match_temperature, match_exposure, match_filter = False, False, True

  new_stacks = []
  for run in stack.runs:
    light_frames = run.frames
    if debug:
      print(f'run light_frames is {light_frames}')
    assert light_frames, f'light_frames is {light_frames}'
    assert all(
        file_io.is_light_frame(fits_headers[f]) and
        not file_io.is_master(fits_headers[f]) for f in light_frames)
    light_runs = runs.get_runs(light_frames, fits_headers, context.max_run_gap,
                               context.min_frames_in_run)
    if debug:
      print(f'light_runs is {light_runs}')
    assert len(light_runs) == 1, f'light_runs are {light_runs}'
    light_run = light_runs[0]

    if debug:
      print(f'Associating {image_type} calibration frames to '
            f'run of {len(light_frames)} light frames')

    # Find all of the calibration frames in the library that match the given
    # image_type and camera. If there are no such frames, end early.
    cal_frames = [
        f for f, header in fits_headers.items()
        if header and not file_io.is_master(header) and 'IMAGETYP' in header and
        'INSTRUME' in header and header['IMAGETYP'] == image_type and
        header['INSTRUME'] == stack.camera
    ]
    if not cal_frames:
      if debug:
        print(f'WARNING: Could not find any {image_type} calibration frames')
      continue

    if debug:
      print(f'Selected {len(cal_frames)} {image_type} frames for camera '
            f'{stack.camera}')

    # Maybe select the calibration frames that have exactly the same exposure
    # as the given light frames, or the next longest exposure. If there are no
    # such frames, end early.
    if match_exposure:
      exposure = fits_headers[light_frames[0]]['EXPOSURE']
      assert all(fits_headers[f]['EXPOSURE'] == exposure for f in light_frames)
      unique_exposures = set(fits_headers[f]['EXPOSURE'] for f in cal_frames)
      selected_exposure = next(e for e in unique_exposures if e >= exposure)
      cal_frames = [
          f for f in cal_frames if 'EXPOSURE' in fits_headers[f] and
          fits_headers[f]['EXPOSURE'] == selected_exposure
      ]
      if not cal_frames:
        if debug:
          print(f'WARNING: Could not find any calibration frames '
                f'for exposure {exposure}')
        continue

      if debug:
        print(f'Selected {len(cal_frames)} {image_type} frames for '
              'exposure {exposure}')

    # Maybe select the calibration frames that have the same filter as the light
    # frames. If there are no such frames, end early.
    if match_filter:
      filter = fits_headers[light_frames[0]]['FILTER']
      assert all(fits_headers[f]['FILTER'] == filter for f in light_frames)
      cal_frames = [
          f for f in cal_frames
          if 'FILTER' in fits_headers[f] and fits_headers[f]['FILTER'] == filter
      ]
      if not cal_frames:
        if debug:
          print(f'WARNING: Could not find any calibration frames '
                f'for filter {filter}')
        continue

      if debug:
        print(f'Selected {len(cal_frames)} {image_type} frames for '
              f'filter {filter}')

    # Maybe select the calibration frames that have temperatures that are from
    # the same approximate distribution as the light frames. If there are no
    # such frames, end early.
    if match_temperature:
      light_frame_temps = [(f, float(fits_headers[f]['CCD-TEMP']))
                           for f in light_frames
                           if 'CCD-TEMP' in fits_headers[f]]
      mean_light_frame_temp = np.mean([x[1] for x in light_frame_temps])
      cal_frame_temps = [(f, fits_headers[f]['CCD-TEMP']) for f in cal_frames]
      cal_frames = clustering.items_in_distribution(light_frame_temps,
                                                    cal_frame_temps,
                                                    slop=5)
      cal_frames = [f for f, _ in cal_frames]
      if not cal_frames:
        if debug:
          print(f'WARNING: Could not find any calibration frames '
                f'for mean temperature {mean_light_frame_temp:.3f} deg C')
        continue

      if debug:
        print(f'Selected {len(cal_frames)} {image_type} frames by temperature '
              f'(average {mean_light_frame_temp:.3f} deg C)')

    # Select the run of calibration frames that are closest in time to the
    # light frames.
    cal_runs = runs.get_runs(cal_frames, fits_headers, context.max_run_gap,
                             context.min_frames_in_run)
    if debug:
      print(f'cal_runs is {cal_runs}')
    if not len(cal_runs):
      if debug:
        print(
            f'WARNING: Could not find any acceptable runs of calibration frames'
        )
      continue

    cal_run = cal_runs[np.argmin(
        [runs.run_time_delta(light_run, cal_run) for cal_run in cal_runs])]
    cal_frames = runs.get_frames_for_run(cal_run, cal_frames, fits_headers)
    if debug:
      print(f'Selected {len(cal_frames)} {image_type} frames by run')

    # Create a new Stack containing one Run of calibration frames. If the same
    # Stack is found multiple times, it will be automatically de-duped by its
    # content hash.
    if len(cal_frames) < context.min_frames_in_stack:
      print(f'Run is too short; ignoring it')
      continue

    new_run = Run(frames=[FrameHash(h) for h in cal_frames],
                  stack_dependencies=copy(stack_dependencies))
    assert None not in new_run.stack_dependencies
    new_runs = [new_run]
    new_stack = Stack(
        content_hash=stack_identification._compute_content_hash(new_runs),
        runs=new_runs,
        camera=stack.camera,
        width_pixels=stack.width_pixels,
        height_pixels=stack.height_pixels)

    # Populate details.
    if match_temperature:
      new_stack.mean_temperature = np.mean(
          [fits_headers[f]['CCD-TEMP'] for f in cal_frames])

    if match_exposure:
      new_stack.exposure_s = fits_headers[cal_frames[0]]['EXPOSURE']

    if match_filter:
      new_stack.filter = fits_headers[cal_frames[0]]['FILTER']

    # Add the new calibration Stack as a dependency to the given Stack.
    run.stack_dependencies.append(new_stack.content_hash)
    new_stacks.append(new_stack)

  return stack, new_stacks


@ray.remote
def associate_calibration_frames(context):
  print(f'Associating calibration frames...')

  fits_headers = ray.get(context.fits_headers)
  stacks = ray.get(context.stacks)

  def assert_frame_in_stacks(stacks, debug_hash):
    found = False
    for stack in stacks:
      assert type(stack) is Stack
      for run in stack.runs:
        if debug_hash in run.frames:
          found = True
          break

    assert found, f'Debug hash {debug_hash} not found in stacks'
    print(f'### Debug hash {debug_hash} was found in stacks')

  debug_hash = 'cbc3b0f593d8a78f'
  assert_frame_in_stacks(stacks, debug_hash)

  # At this point, stacks is just a list of stacks. The final content hashes
  # are not known yet, because the calibration frames aren't associated yet.
  # TODO: Maybe parallelize this if it's slow.
  output_stacks = []
  for stack in stacks:
    stack, bias_stacks = _associate(context, stack, 'bias', fits_headers)
    for bias_stack in bias_stacks:
      if bias_stack not in output_stacks:
        output_stacks.append(bias_stack)

    stack, dark_stacks = _associate(context, stack, 'dark', fits_headers)
    for dark_stack in dark_stacks:
      if dark_stack not in output_stacks:
        output_stacks.append(dark_stack)

    stack_dependencies = list(
        set(s.content_hash for s in bias_stacks + dark_stacks))
    assert None not in stack_dependencies
    stack, flat_stacks = _associate(context,
                                    stack,
                                    'flat',
                                    fits_headers,
                                    stack_dependencies=stack_dependencies)
    for flat_stack in flat_stacks:
      if flat_stack not in output_stacks:
        output_stacks.append(flat_stack)

    for run in stack.runs:
      assert len(run.stack_dependencies) == len(set(run.stack_dependencies))

    assert stack not in output_stacks
    output_stacks.append(stack)

  assert_frame_in_stacks(output_stacks, debug_hash)

  def print_stack(prefix, stack):
    print(f'### {prefix} stack {stack}')
    for run in stack.runs:
      print(f'### {prefix} run {run}')

  stacks = {}
  content_hashes = []
  geometric_hashes = []
  for stack in output_stacks:
    assert stack.content_hash != None

    if stack.geometric_hash != None:
      if stack.geometric_hash in stacks:
        print_stack('new: ', stack)
        print_stack('previous: ', stacks[stack.geometric_hash])
        assert False, (
            f'### stack {stack} has non-unique geometric hash; previous stack '
            f'was {stacks[stack.geometric_hash]}')
      geometric_hashes.append(stack.geometric_hash)
      stacks[stack.geometric_hash] = stack
      continue

    if stack.content_hash in stacks:
      print_stack('new: ', stack)
      print_stack('previous: ', stacks[stack.content_hash])
      assert False, (
          f'### stack {stack} has non-unique content hash; previous stack was '
          f'{stacks[stack.content_hash]}')
    content_hashes.append(stack.content_hash)
    stacks[stack.content_hash] = stack

  print(f'### There are {len(content_hashes)} unique content hashes')
  print(f'### There are {len(geometric_hashes)} unique geometric hashes')

  # TODO: It's a bit weird to have to include both kinds of hashes here.
  # stacks = {s.geometric_hash or s.content_hash: s for s in output_stacks}
  # assert len(stacks) == len(
  #     output_stacks), f'{len(output_stacks)} stacks became {len(stacks)} stacks'

  context.stacks = ray.put(stacks)
  ray_diskcache.set('stacks', stacks)
  assert_frame_in_stacks(stacks.values(), debug_hash)
  print(f'### Wrote {len(stacks)} stacks into cache')

  return context, None
