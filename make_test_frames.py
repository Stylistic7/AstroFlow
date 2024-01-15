import os
import astropy.io.fits as pyfits
import numpy as np
import datetime
import random
import xxhash
import uuid

from plate_solving import PlateSolution


def _randomize_dict(input):
  randomized = {}
  for k, v in sorted(input.items(), key=lambda x: random.random()):
    randomized[k] = v

  return randomized


def _add_dither(center, max_distance_deg):
  ra, dec = center
  distance = np.random.uniform(0.0, max_distance_deg)
  angle = np.random.uniform(0.0, 2.0 * np.pi)

  return ra + distance * np.sin(angle), dec + distance * np.cos(angle)


def make_test_frames(start_datetime=datetime.datetime(2023, 12, 6, 16, 29, 43),
                     number=10,
                     image_type='Light',
                     prefix=None,
                     ids=None,
                     project_dir=None,
                     ra=100,
                     dec=45,
                     centers=None,
                     dither=0.01,
                     focal_length_m=0.35,
                     pixel_pitch_um=3.76,
                     width_pixels=10800,
                     height_pixels=7200,
                     exposure=300,
                     filter='H',
                     fill_value=None,
                     real_hash=False,
                     num_stars=100,
                     image_scale=1.0,
                     rotation_angle_deg=90.0,
                     image_hashes=None,
                     fits_headers=None,
                     plate_solutions=None):
  if not prefix:
    prefix = str(uuid.uuid4())

  number = len(ids) if ids else number

  ras = ra if type(ra) is list else [ra] * number
  decs = dec if type(dec) is list else [dec] * number
  ids = ids if ids else [f'{prefix}_{i:05}.fit' for i in range(number)]
  centers = centers if centers else [(ra, dec)] * number

  if not image_hashes:
    image_hashes = {}

  if not fits_headers:
    fits_headers = {}

  if not plate_solutions:
    plate_solutions = {}

  for i in range(number):
    fits_filename = ids[i]
    if project_dir:
      fits_filename = os.path.join(project_dir, fits_filename)

    if fill_value:
      image_data = np.ones((100, 100)) * fill_value
    else:
      image_data = np.random.random((100, 100))

    if real_hash:
      image_hash = xxhash.xxh128(image_data.tobytes()).hexdigest()
    else:
      image_hash = 'hash_for_' + fits_filename

    image_hashes[image_hash] = fits_filename

    center = _add_dither(centers[i], dither)

    date_obs = start_datetime + i * datetime.timedelta(seconds=exposure + 10)

    deg_per_pix = image_scale / 3600.0

    fits_header = {
        'SIMPLE': True,
        'BITPIX': 16,
        'DATE-OBS': date_obs.strftime("%Y-%m-%dT%H:%M:%S"),
        'RA': center[0],
        'DEC': center[1],
        'IMAGETYP': image_type,
        'EXPOSURE': exposure,
        'CTYPE1': 'RA---TAN-SIP',
        'CTYPE2': 'DEC--TAN-SIP',
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CRVAL1': center[0],
        'CRVAL2': center[1],
        'CRPIX1': width_pixels / 2 + 0.5,
        'CRPIX2': height_pixels / 2 + 0.5,
        'CDELT1': deg_per_pix,
        'CDELT2': deg_per_pix,
        'CD1_1': deg_per_pix * np.cos(rotation_angle_deg),
        'CD1_2': -deg_per_pix * np.sin(rotation_angle_deg),
        'CD2_1': deg_per_pix * np.sin(rotation_angle_deg),
        'CD2_2': deg_per_pix * np.cos(rotation_angle_deg),
        'WCSAXES': 2,
        'NAXIS': 2,
        'NAXIS1': width_pixels,
        'NAXIS2': height_pixels,
        'IMAGEW': width_pixels,
        'IMAGEH': height_pixels,
        'FILTER': filter,
        'FOCALLEN': focal_length_m,
        'XPIXSZ': pixel_pitch_um,
        'YPIXSZ': pixel_pitch_um,
    }
    fits_headers[image_hash] = fits_header

    if image_type == 'Light':
      plate_solutions[image_hash] = PlateSolution(
          image_filename=fits_filename,
          header=fits_header,
          center=center,
          image_scale=image_scale,
          rotation_angle=rotation_angle_deg)

    if not project_dir:
      continue

    hdu = pyfits.PrimaryHDU(image_data)
    for k, v in fits_header.items():
      hdu.header[k] = v
    hdu.writeto(fits_filename)

  return _randomize_dict(image_hashes), _randomize_dict(
      fits_headers), _randomize_dict(plate_solutions)
