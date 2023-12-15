import os
import astropy.io.fits as pyfits
import numpy as np
import uuid
import datetime
import random

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


def _footprint_around_center(center, rotation_angle_deg, width_deg,
                             height_deg):
  center = np.array(center)

  # A `rotation_angle_deg` of 0 means the long axis (width) of the image is
  # parallel to lines of constant RA.
  theta = np.radians(rotation_angle_deg)
  half_width_vec = np.array(
      [width_deg * np.cos(theta) / 2, width_deg * np.sin(theta) / 2])
  half_height_vec = np.array(
      [height_deg * np.sin(theta) / 2, height_deg * np.cos(theta) / 2])

  return [
      tuple(center - half_width_vec - half_height_vec),
      tuple(center + half_width_vec - half_height_vec),
      tuple(center + half_width_vec + half_height_vec),
      tuple(center - half_width_vec + half_height_vec)
  ]


def make_test_frames(start_datetime=datetime.datetime(2023, 12, 6, 16, 29, 43),
                     number=10,
                     image_type='Light',
                     prefix=None,
                     ids=None,
                     project_dir=None,
                     ra=100,
                     dec=45,
                     centers=None,
                     ra_span=4,
                     dec_span=2,
                     dither=0.01,
                     exposure=300,
                     num_stars=100,
                     image_scale=1.0,
                     rotation_angle_deg=90.0):
  if not prefix:
    prefix = str(uuid.uuid4())

  number = len(ids) if ids else number

  ras = ra if type(ra) is list else [ra] * number
  decs = dec if type(dec) is list else [dec] * number
  ids = ids if ids else [f'{prefix}_{i:05}.fit' for i in range(number)]
  centers = centers if centers else [(ra, dec)] * number

  fits_headers = {}
  plate_solutions = {}
  for i in range(number):
    fits_filename = ids[i]
    if project_dir:
      fits_filename = os.path.join(project_dir, fits_filename)

    center = _add_dither(centers[i], dither)

    fits_header = {
        'DATE-OBS':
        start_datetime + i * datetime.timedelta(seconds=exposure + 10),
        'IMAGETYP': image_type,
        'EXPOSURE': exposure,
        'CRVAL1': center[0],
        'CRVAL2': center[1]
    }
    fits_headers[fits_filename] = fits_header

    if image_type == 'Light':
      plate_solutions[fits_filename] = PlateSolution(
          image_filename=fits_filename,
          header={
              'CRVAL1': center[0],
              'CRVAL2': center[1]
          },
          center=center,
          image_scale=image_scale,
          rotation_angle=rotation_angle_deg,
          footprint=_footprint_around_center(center, rotation_angle_deg,
                                             ra_span, dec_span))

    if not project_dir:
      continue

    image_data = np.arange(100.0)
    hdu = pyfits.PrimaryHDU(image_data)
    for k, v in fits_header.items():
      hdu.header[k] = v
    hdu.writeto(fits_filename)

  return _randomize_dict(fits_headers), _randomize_dict(plate_solutions)
