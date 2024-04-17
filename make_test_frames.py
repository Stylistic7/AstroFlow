import os
import astropy.io.fits as pyfits
import numpy as np
from scipy.signal.windows import gaussian
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
    ],
    'Orion Pane 1': [
        'testdata/Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-210506_0001.fit',
        'testdata/Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-211007_0002.fit',
        'testdata/Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-211510_0003.fit',
    ],
    'Orion Pane 2': [
        'testdata/Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220301_0001.fit',
        'testdata/Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220803_0002.fit',
        'testdata/Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-221305_0003.fit',
    ],
    'Orion Pane 3': [
        'testdata/Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-230116_0001.fit',
        'testdata/Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-230618_0002.fit',
        'testdata/Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-231120_0003.fit',
    ],
    'Orion Pane 4': [
        'testdata/Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-001210_0001.fit',
        'testdata/Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-001712_0002.fit',
        'testdata/Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-002215_0003.fit',
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


def _make_flat_frame(height_pixels, width_pixels, std=None):
  if not std:
    std = width_pixels / 2.0

  gkern1d = gaussian(width_pixels, std=std).reshape(width_pixels, 1)
  img = np.outer(gkern1d, gkern1d)

  bottom = int((width_pixels - height_pixels) / 2)
  top = bottom + height_pixels
  img = img[bottom:top, 0:width_pixels]

  img *= 32768

  return img.astype(np.uint16)


def copy_testdata_frames(target_name,
                         dest_dir,
                         prefix='',
                         fits_header_overrides=None,
                         start_datetime=None):
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
          date_obs += datetime.timedelta(seconds=int(header['EXPOSURE']) + 1)

        fits.flush()
        fits.close()


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

    if image_type == 'Flat':
      image_data = _make_flat_frame(height_pixels, width_pixels)
    elif fill_value:
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
    fits_headers[image_hash] = file_io._normalize_header(fits_header)

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
