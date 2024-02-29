import ray
import numpy as np
from copy import copy

import file_io
import ray_diskcache
import runs
import clustering
from stack import Stack, Run, FrameHash


def _associate(stack,
               image_type,
               fits_headers,
               stack_dependencies=None,
               debug=False):
  if debug:
    print(f'### Entering _associate for {stack}')
  if not stack_dependencies:
    stack_dependencies = []

  if image_type == 'Bias':
    match_temperature, match_exposure, match_filter = True, False, False
  elif image_type == 'Dark':
    match_temperature, match_exposure, match_filter = True, True, False
  elif image_type == 'Flat':
    match_temperature, match_exposure, match_filter = False, False, True

  new_stacks = []
  for run in stack.runs:
    light_frames = run.frames
    assert all(file_io.is_light_frame(fits_headers[f]) for f in light_frames)
    light_runs = runs.get_runs(light_frames, fits_headers)
    assert len(light_runs) == 1
    light_run = light_runs[0]

    if debug:
      print(f'### Associating {image_type} calibration frames to '
            f'run of {len(light_frames)} light frames')

    # Find all of the calibration frames in the library that match the given
    # image_type and camera. If there are no such frames, end early.
    cal_frames = [
        f for f, header in fits_headers.items()
        if not file_io.is_master(header) and 'IMAGETYP' in header and
        'INSTRUME' in header and header['IMAGETYP'] == image_type and
        header['INSTRUME'] == stack.camera
    ]
    if not cal_frames:
      if debug:
        print(f'WARNING: Could not find any {image_type} calibration frames')
      continue

    if debug:
      print(f'### Selected {len(cal_frames)} {image_type} frames for camera '
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
        print(f'### Selected {len(cal_frames)} {image_type} frames by exposure')

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
        print(f'### Selected {len(cal_frames)} {image_type} frames by filter')

    # Maybe select the calibration frames that have temperatures that are from
    # the same approximate distribution as the light frames. If there are no
    # such frames, end early.
    if match_temperature:
      light_frame_temps = [(f, float(fits_headers[f]['CCD-TEMP']))
                           for f in light_frames
                           if 'CCD-TEMP' in fits_headers[f]]
      cal_frame_temps = [(f, fits_headers[f]['CCD-TEMP']) for f in cal_frames]
      cal_frames = clustering.items_in_distribution(light_frame_temps,
                                                    cal_frame_temps,
                                                    slop=5)
      cal_frames = [f for f, _ in cal_frames]
      if not cal_frames:
        if debug:
          print(
              f'WARNING: Could not find any calibration frames '
              f'for mean temperature {np.mean([x[1] for x in light_frame_temps])}'
          )
        continue

      if debug:
        print(
            f'### Selected {len(cal_frames)} {image_type} frames by temperature'
        )

    # Select the run of calibration frames that are closest in time to the
    # light frames.
    cal_runs = runs.get_runs(cal_frames, fits_headers)
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
      print(f'### Selected {len(cal_frames)} {image_type} frames by run')

    # Create a new Stack containing one Run of calibration frames. If the same
    # Stack is found multiple times, it will be automatically de-duped by its
    # content hash.
    new_run = Run(frames=[FrameHash(h) for h in cal_frames],
                  stack_dependencies=copy(stack_dependencies))
    new_stack = Stack(runs=[new_run],
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
    run.stack_dependencies.append(new_stack.content_hash())
    new_stacks.append(new_stack)

  return stack, new_stacks


@ray.remote
def associate_calibration_frames(context):
  print(f'Associating calibration frames...')

  fits_headers = ray.get(context.fits_headers)
  stacks = ray.get(context.stacks)

  # At this point, stacks is just a list of stacks. The final content hashes
  # are not known yet, because the calibration frames aren't associated yet.
  # TODO: Maybe parallelize this if it's slow.
  output_stacks = []
  for stack in stacks:
    stack, bias_stacks = _associate(stack, 'Bias', fits_headers)
    output_stacks.extend(bias_stacks)

    stack, dark_stacks = _associate(stack, 'Dark', fits_headers)
    output_stacks.extend(dark_stacks)

    stack_dependencies = list(
        set(s.content_hash() for s in bias_stacks + dark_stacks))
    stack, flat_stacks = _associate(stack,
                                    'Flat',
                                    fits_headers,
                                    stack_dependencies=stack_dependencies)
    output_stacks.extend(flat_stacks)

    for run in stack.runs:
      assert len(run.stack_dependencies) == len(set(run.stack_dependencies))

    output_stacks.append(stack)

  stacks = {s.hash(): s for s in output_stacks}

  ray_diskcache.set('stacks', stacks)
  context.stacks = ray.put(stacks)
  return context
