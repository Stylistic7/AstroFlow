import file_io
import bisect
import astropy.io.fits as pyfits
import datetime
import pytz
import numpy as np
import itertools
import structlog

import runs
from cache import cache
from stack import Stack, Status

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


# TODO: Move to encoding FLATVERS in the symlinked file names, vs. modifying the
# FITS header.
def set_flat_versions(stack, fits_headers):
  print(f'Setting FLATVERS for {len(stack.flat_frame_associations)} '
        f'associations')
  for flat_version, (light_run, flat_run) in enumerate(
      stack.flat_frame_associations):
    light_frames = runs.get_frames_for_run(light_run, stack.light_frames,
                                           fits_headers)
    flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
                                          fits_headers)
    _set_flatvers_fits_header(light_frames + flat_frames, flat_version)


def run_distance(a, b):
  start_a, end_a = a
  midpoint_a = start_a + (end_a - start_a) / 2

  start_b, end_b = b
  midpoint_b = start_b + (end_b - start_a) / 2

  return abs(midpoint_a - midpoint_b)


# TODO: Move this into a new file.
def associate_flat_frames_for_stack(stack, flat_frames, fits_headers):
  imaging_runs = runs.get_runs(stack.light_frames, fits_headers)
  flat_runs = runs.get_runs(flat_frames, fits_headers)

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

  stack.flat_frame_associations = flat_frame_associations

  stack.flat_frames = runs.get_frames_for_runs(
      (flat_run for _, flat_run in flat_frame_associations), flat_frames,
      fits_headers)
