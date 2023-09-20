import file_io
import bisect
import astropy.io.fits as pyfits

from cache import cache


def _frames_for_date(frames, date):
  return [f for f in frames if file_io.read_fits(f)['DATE'] == date]


def _set_flatvers_fits_header(filenames, flat_version):
  for filename in filenames:
    with pyfits.open(filename, mode='update') as fits:
      # print(f'Setting FLATVERS = {flat_version} for file {filename}')
      fits[0].header['FLATVERS'] = flat_version


def _dates_to_strings(associations):
  output = {}
  for k, v in associations.items():
    output[str(k)] = str(v)

  return output


def _associate_dates(light_frame_dates, flat_frame_dates):
  associations = {}
  for flat_version, ffd in enumerate(flat_frame_dates):
    for lfd in light_frame_dates:
      if lfd >= ffd:
        associations[lfd] = flat_version

  for lfd in light_frame_dates:
    if lfd not in associations:
      associations[lfd] = 0

  return associations


def filter_flat_frames(flat_frames, rig, filter, camera_configs):
  output = []
  for camera, _, _ in camera_configs:
    output.extend([
        f for f in flat_frames if file_io.fits_matches(
            f, ['INSTRUME', 'RIG', 'FILTER'], [camera, rig, filter])
    ])

  return list(sorted(set(output)))


def run(light_frames, flat_frames):
  fits_headers = cache['fits_headers']

  light_frame_dates = sorted(set(
      fits_headers[f]['DATE'] for f in light_frames))
  flat_frame_dates = sorted(set(fits_headers[f]['DATE'] for f in flat_frames))

  return _dates_to_strings(
      _associate_dates(light_frame_dates, flat_frame_dates))


def set_flat_versions(light_frames, flat_frames, associations):
  light_frame_dates = list(
      sorted(set(file_io.read_fits(f)['DATE'] for f in light_frames)))
  flat_frame_dates = list(
      sorted(set(file_io.read_fits(f)['DATE'] for f in flat_frames)))

  for flat_version, date in enumerate(flat_frame_dates):
    frames_for_date = _frames_for_date(flat_frames, date)
    _set_flatvers_fits_header(frames_for_date, flat_version)
    print(f'Set FLATVERS={flat_version} for {len(frames_for_date)} flat frames')

  for date, flat_version in associations.items():
    frames_for_date = _frames_for_date(light_frames, date)
    _set_flatvers_fits_header(frames_for_date, flat_version)
    print(
        f'Set FLATVERS={flat_version} for {len(frames_for_date)} light frames')