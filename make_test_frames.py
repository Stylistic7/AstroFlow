import os
import astropy.io.fits as pyfits
import numpy as np
import datetime
import random
import uuid
import shutil
import gzip

import plate_solving
import file_io

_TESTDATA_TARGETS = {
    'M42': [
        'testdata/Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit',
        'testdata/Light_M42_180.0s_Bin1_2600MM_20231130-234248_0024.fit',
        'testdata/Light_M42_180.0s_Bin1_2600MM_20231130-234549_0025.fit',
    ],
    'NGC7635': [
        'testdata/Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213019_0004.fit',
        'testdata/Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213521_0005.fit',
        'testdata/Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-214053_0006.fit',
    ]
}


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


def copy_testdata_frames(target_name,
                         dest_dir,
                         prefix='',
                         fits_header_overrides=None,
                         start_datetime=None,
                         image_hashes=None,
                         fits_headers=None,
                         plate_solutions=None):
  if not image_hashes:
    image_hashes = {}

  if not fits_headers:
    fits_headers = {}

  if not plate_solutions:
    plate_solutions = {}

  if not fits_header_overrides:
    fits_header_overrides = {}

  if not prefix:
    prefix = target_name

  date_obs = start_datetime
  for filename in _TESTDATA_TARGETS[target_name]:
    dest_filename = os.path.join(dest_dir, f'{prefix}_{str(uuid.uuid4())}.fit')
    with gzip.open(filename + '.gz', 'rb') as f_in:
      with open(dest_filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

        fits = pyfits.open(dest_filename, mode='update')
        header = fits[0].header
        image = fits[0].data

        # Add a small amount of noise so it will have a unique hash.
        assert image.dtype == np.uint16
        image += np.random.randint(low=0,
                                   high=2,
                                   size=image.shape,
                                   dtype=image.dtype)

        if date_obs:
          header['DATE-OBS'] = date_obs.strftime("%Y-%m-%dT%H:%M:%S")
          print('### copy_testdata_frames overwrote DATE-OBS to',
                header['DATE-OBS'])
          date_obs += datetime.timedelta(seconds=int(header['EXPOSURE']) + 1)

        fits.flush()
        fits.close()

    hash = file_io.hash_of_image_data(dest_filename)
    image_hashes[hash] = dest_filename

    header = file_io.read_header(dest_filename)
    print('### copy_testdata_frames DATE-OBS on disk is', header['DATE-OBS'])
    header.update(fits_header_overrides)

    fits_headers[hash] = header
    plate_solutions[hash] = plate_solving.solve_field(dest_filename)

  return _randomize_dict(image_hashes), _randomize_dict(
      fits_headers), _randomize_dict(plate_solutions)


def make_synthetic_frames(start_datetime=datetime.datetime(
    2023, 12, 6, 16, 29, 43),
                          number=1,
                          image_type='Light',
                          prefix=None,
                          ids=None,
                          dest_dir=None,
                          ra=100,
                          dec=45,
                          centers=None,
                          dither=0.01,
                          focal_length_m=0.35,
                          pixel_pitch_um=3.76,
                          instrument='ASI6200MM Pro',
                          width_pixels=1080,
                          height_pixels=720,
                          temperature=-20,
                          exposure=300,
                          filter='H',
                          fill_value=None,
                          num_stars=100,
                          image_scale=1.0,
                          rotation_angle_deg=90.0,
                          image_hashes=None,
                          fits_headers=None,
                          plate_solutions=None):
  if not prefix:
    prefix = str(uuid.uuid4())

  number = len(ids) if ids else number

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
    if dest_dir:
      fits_filename = os.path.join(dest_dir, fits_filename)

    if fill_value:
      image_data = np.ones(
          (height_pixels, width_pixels), dtype=np.uint16) * fill_value
    else:
      image_data = np.random.random(
          (height_pixels, width_pixels)).astype(np.uint16)

    # Add a small amount of noise so it will have a unique hash.
    assert image_data.dtype == np.uint16
    image_data += np.random.randint(low=0,
                                    high=2,
                                    size=image_data.shape,
                                    dtype=image_data.dtype)

    image_hash = 'hash_for_' + fits_filename
    image_hashes[image_hash] = fits_filename

    center = _add_dither(centers[i], dither)

    date_obs = start_datetime + i * datetime.timedelta(seconds=exposure + 1)

    deg_per_pix = image_scale / 3600.0

    fits_header = {
        'SIMPLE': True,
        'BITPIX': 16,
        'DATE-OBS': date_obs.strftime("%Y-%m-%dT%H:%M:%S"),
        'RA': center[0],
        'DEC': center[1],
        'IMAGETYP': image_type,
        'INSTRUME': instrument,
        'CCD-TEMP': temperature,
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
      plate_solutions[image_hash] = plate_solving.PlateSolution(
          image_filename=fits_filename,
          header=fits_header,
          num_stars=num_stars,
          center=center,
          image_scale=image_scale,
          rotation_angle=rotation_angle_deg)

    if not dest_dir:
      continue

    hdu = pyfits.PrimaryHDU(image_data)
    for k, v in fits_header.items():
      hdu.header[k] = v
    hdu.writeto(fits_filename)

  return _randomize_dict(image_hashes), _randomize_dict(
      fits_headers), _randomize_dict(plate_solutions)
