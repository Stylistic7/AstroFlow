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


def _set_flatvers_fits_header(hashes, image_hashes, flat_version):
  for hash in hashes:
    fits = pyfits.open(image_hashes[hash], mode='update')
    fits[0].header['FLATVERS'] = flat_version
    fits.close()


def _dates_to_strings(associations):
  output = {}
  for k, v in associations.items():
    output[str(k)] = str(v)

  return output


# TODO: Move to encoding FLATVERS in the symlinked file names, vs. modifying the
# FITS header.
def set_flat_versions(stack, image_hashes, fits_headers):
  print(f'Setting FLATVERS for {len(stack.flat_frame_associations)} '
        f'associations')
  for flat_version, (light_run,
                     flat_run) in enumerate(stack.flat_frame_associations):
    light_frames = runs.get_frames_for_run(light_run, stack.light_frames,
                                           fits_headers)
    flat_frames = runs.get_frames_for_run(flat_run, stack.flat_frames,
                                          fits_headers)
    _set_flatvers_fits_header(light_frames + flat_frames, image_hashes,
                              flat_version)
