import file_io
import bisect
import astropy.io.fits as pyfits
import datetime
import pytz
import numpy as np
import itertools
import structlog

from cache import cache
from stack import Stack, Status, FileList

logger = structlog.get_logger()
log = logger


def _set_flatvers_fits_header(filenames, flat_version):
  for filename in filenames:
    with pyfits.open(filename, mode='update') as fits:
      fits[0].header['FLATVERS'] = flat_version


def _dates_to_strings(associations):
  output = {}
  for k, v in associations.items():
    output[str(k)] = str(v)

  return output


def set_flat_versions(stack, fits_headers):
  print(f'Setting FLATVERS for {len(stack.flat_frame_associations)} '
        f'associations')
  for flat_version, (light_frames, flat_frames) in enumerate(
      stack.flat_frame_associations):
    _set_flatvers_fits_header(light_frames + flat_frames, flat_version)


# TODO: Move this into a new file.
def get_runs(frames, fits_headers):
  if not frames: return []

  # print(f'### get_runs for {len(frames)} frames')
  frames = sorted(frames, key=lambda f: fits_headers[f]['DATE-OBS'])

  # print('### sorted light frame dates:')
  # for light_frame in stack.light_frames:
  #   print('###', type(fits_headers[light_frame]['DATE-OBS']))
  #   print('###', fits_headers[light_frame]['DATE-OBS'])

  max_gap = datetime.timedelta(hours=1)
  runs = []
  prev_date_obs = pytz.utc.localize(datetime.datetime.min)
  run_start = None
  frame_count = 0
  for frame in frames:
    current_date_obs = fits_headers[frame]['DATE-OBS']
    if current_date_obs > prev_date_obs + max_gap:
      # Save the current run, if there is one.
      if run_start:
        log.debug(f'### Appending run with run start {run_start} '
                  f'and frame count {frame_count}')
        assert frame_count
        runs.append((run_start, prev_date_obs))

      # Start a new run.
      run_start = current_date_obs
      frame_count = 1
    else:
      frame_count += 1

    prev_date_obs = current_date_obs

  runs.append((run_start, prev_date_obs))

  return runs


def run_distance(a, b):
  start_a, end_a = a
  midpoint_a = start_a + (end_a - start_a) / 2

  start_b, end_b = b
  midpoint_b = start_b + (end_b - start_a) / 2

  return abs(midpoint_a - midpoint_b)


# TODO: Move this into a new file.
def associate_flat_frames_for_stack(stack, flat_frames, fits_headers):
  print(f'Associating flat frames for stack {stack}')
  imaging_runs = get_runs(stack.light_frames, fits_headers)
  # for start, end in imaging_runs:
  #   print(f'imaging run start {start} end {end}')

  flat_runs = get_runs(flat_frames, fits_headers)
  # for start, end in flat_runs:
  #   print(f'### flat run start {str(start)} end {str(end)}')

  flat_frame_associations = []
  if flat_runs:
    for imaging_run in imaging_runs:
      im_start, im_end = imaging_run
      log.debug(f'### imaging run {str(im_start)} to {str(im_end)}')
      run_distances = [
          run_distance(imaging_run, flat_run) for flat_run in flat_runs
      ]
      index = np.argmin(run_distances)
      flat_frame_associations.append((imaging_run, flat_runs[index]))
      for i, flat_run in enumerate(flat_runs):
        flat_start, flat_end = flat_run
        if i == index:
          log.debug(f'### --> flat run {str(flat_start)} to {str(flat_end)}')
        else:
          log.debug(f'###     flat run {str(flat_start)} to {str(flat_end)}')

  for imaging_run, flat_run in flat_frame_associations:
    log.debug(f'associated imaging run {imaging_run} '
              f'with flat run {flat_run}')

  def is_in_run(frame, run):
    run_start, run_end = run
    date_obs = fits_headers[frame]['DATE-OBS']
    result = date_obs >= run_start and date_obs <= run_end
    return result

  stack.flat_frame_associations = []
  for imaging_run, flat_run in flat_frame_associations:
    log.debug(f'### finding light frames in imaging_run {imaging_run}')
    lights = list(f for f in stack.light_frames if is_in_run(f, imaging_run))
    log.debug(f'### finding flat frames in flat_run {flat_run}')
    flats = list(f for f in flat_frames if is_in_run(f, flat_run))
    assert lights
    assert flats
    stack.flat_frame_associations.append((lights, flats))

  stack.flat_frames = FileList(
      itertools.chain.from_iterable(
          flat_frames for _, flat_frames in stack.flat_frame_associations))
