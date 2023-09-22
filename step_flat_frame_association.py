import file_io
import bisect
import astropy.io.fits as pyfits

from cache import cache


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
