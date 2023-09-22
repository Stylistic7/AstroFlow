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
  flat_runs = list(
      set(flat_run for _, flat_run in stack.flat_frame_associations))

  print(f'Setting FLATVERS for {len(stack.flat_frame_associations)} '
        f'associations')
  for flat_version, ((imaging_run_start, imaging_run_end),
                     (flat_run_start, flat_run_end)) in enumerate(
                         stack.flat_frame_associations):
    light_frames_in_run = [
        f for f in stack.light_frames
        if fits_headers[f]['DATE-OBS'] >= imaging_run_start and
        fits_headers[f]['DATE-OBS'] <= imaging_run_end
    ]
    print(f'Found {len(light_frames_in_run)} in imaging run')

    flat_frames_in_run = [
        f for f in stack.flat_frames
        if fits_headers[f]['DATE-OBS'] >= flat_run_start and
        fits_headers[f]['DATE-OBS'] <= flat_run_end
    ]
    print(f'Found {len(flat_frames_in_run)} in flat run')

    print(f'Setting imaging run {imaging_run_start} to {imaging_run_end} '
          f'to flat run {flat_run_start} to {flat_run_end} with '
          f'FLATVERS={flat_version}')
    _set_flatvers_fits_header(light_frames_in_run + flat_frames_in_run,
                              flat_version)
