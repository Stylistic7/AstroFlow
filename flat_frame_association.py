import file_io
import bisect
import astropy.io.fits as pyfits


def frames_for_date(frames, date):
  return [f for f in frames if file_io.read_fits(f)['DATE'] == date]


def set_flatvers_fits_header(filenames, flat_version):
  for filename in filenames:
    with pyfits.open(filename, mode='update') as fits:
      # print(f'Setting FLATVERS = {flat_version} for file {filename}')
      fits[0].header['FLATVERS'] = flat_version


def associate_dates(light_frame_dates, flat_frame_dates):
  associations = {}
  for flat_version, ffd in enumerate(flat_frame_dates):
    for lfd in light_frame_dates:
      if lfd >= ffd:
        associations[lfd] = flat_version

  for lfd in light_frame_dates:
    if lfd not in associations:
      associations[lfd] = 0

  return associations


def associate_flat_frames(light_frames, flat_frames):
  light_frame_dates = sorted(
      list(set(file_io.read_fits(f)['DATE'] for f in light_frames)))
  flat_frame_dates = sorted(
      list(set(file_io.read_fits(f)['DATE'] for f in flat_frames)))

  associations = associate_dates(light_frame_dates, flat_frame_dates)

  for flat_version, date in enumerate(flat_frame_dates):
    set_flatvers_fits_header(frames_for_date(flat_frames, date), flat_version)

  for date, flat_version in associations.items():
    set_flatvers_fits_header(frames_for_date(light_frames, date), flat_version)