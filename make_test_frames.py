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


# TODO: Strongly-type the physical units in this file.
def _add_dither(center, max_distance):
  ra, dec = center
  distance = np.random.uniform(0.0, max_distance)  # degrees
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
                     project_dir=None,
                     ra=100,
                     dec=45,
                     ra_span=4,
                     dec_span=2,
                     dither=0.01,
                     exposure=300,
                     num_stars=100,
                     image_scale=1.0,
                     rotation_angle=90.0):
  if not prefix:
    prefix = str(uuid.uuid4())

  fits_headers = {}
  plate_solutions = {}
  for i in range(number):
    fits_filename = f'{prefix}_{i:05}.fit'
    if project_dir:
      fits_filename = os.path.join(project_dir, fits_filename)

    center = _add_dither((ra, dec), dither)

    fits_header = {
        'DATE-OBS':
        start_datetime + i * datetime.timedelta(seconds=exposure + 10),
        'IMAGETYP': image_type,
        'EXPOSURE': exposure,
        'CRVAL1': center[0],
        'CRVAL2': center[1]
    }
    fits_headers[fits_filename] = fits_header

    plate_solutions[fits_filename] = PlateSolution(
        image_filename=fits_filename,
        header={
            'CRVAL1': center[0],
            'CRVAL2': center[1]
        },
        center=center,
        image_scale=image_scale,
        rotation_angle=rotation_angle,
        footprint=_footprint_around_center(center, rotation_angle, ra_span,
                                           dec_span))

    if not project_dir:
      continue

    image_data = np.arange(100.0)
    hdu = pyfits.PrimaryHDU(image_data)
    for k, v in fits_header.items():
      hdu.header[k] = v
    hdu.writeto(fits_filename)

  return _randomize_dict(fits_headers), _randomize_dict(plate_solutions)
