from ray_infra import ray_local

import os
import astropy.io.fits as pyfits  # type: ignore
import numpy as np
from scipy.signal.windows import gaussian  # type: ignore
import datetime
import random
import uuid
import shutil
import xxhash

from dependency_graph import Graph, run_graph
from metadata.classes import ImageMetadata
from metadata.image_metadata import compute_image_metadata, gather_compute_image_metadata
from context import context, WriteMode
import plate_solving
from file_io.normalize_header import normalize_header

_TESTDATA_TARGETS = {
    'M42': [
        'Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit',
        'Light_M42_180.0s_Bin1_2600MM_20231130-234248_0024.fit',
        'Light_M42_180.0s_Bin1_2600MM_20231130-234549_0025.fit',
    ],
    'NGC7635': [
        'Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213019_0004.fit',
        'Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213521_0005.fit',
        'Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-214053_0006.fit',
    ],
    'Orion Pane 1': [
        'Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-210506_0001.fit',
        'Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-211007_0002.fit',
        'Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-211510_0003.fit',
    ],
    'Orion Pane 2': [
        'Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220301_0001.fit',
        'Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-220803_0002.fit',
        'Light_Orion_Pane 2_300.0s_Bin1_6200MM_H_20231222-221305_0003.fit',
    ],
    'Orion Pane 3': [
        'Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-230116_0001.fit',
        'Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-230618_0002.fit',
        'Light_Orion_Pane 3_300.0s_Bin1_6200MM_H_20231222-231120_0003.fit',
    ],
    'Orion Pane 4': [
        'Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-001210_0001.fit',
        'Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-001712_0002.fit',
        'Light_Orion_Pane 4_300.0s_Bin1_6200MM_H_20231223-002215_0003.fit',
    ],
    'Sagittarius Pane 1': [
        'Light_Sag 1_300.0s_Bin1_6200MM_S_20240619-000215_0001.fit',
        'Light_Sag 1_300.0s_Bin1_6200MM_S_20240619-000717_0002.fit',
        'Light_Sag 1_300.0s_Bin1_6200MM_S_20240619-001219_0003.fit',
    ],
    'Sagittarius Pane 2': [
        'Light_Sag 2_300.0s_Bin1_6200MM_S_20240619-010605_0001.fit',
        'Light_Sag 2_300.0s_Bin1_6200MM_S_20240619-011107_0002.fit',
        'Light_Sag 2_300.0s_Bin1_6200MM_S_20240619-011609_0003.fit',
    ],
    'Sagittarius Pane 3': [
        'Light_Sag 3_300.0s_Bin1_6200MM_S_20240619-022800_0001.fit',
        'Light_Sag 3_300.0s_Bin1_6200MM_S_20240619-023301_0002.fit',
        'Light_Sag 3_300.0s_Bin1_6200MM_S_20240619-023803_0003.fit',
    ],
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
                         add_noise=True,
                         fits_header_overrides=None,
                         start_datetime=None):
  if not fits_header_overrides:
    fits_header_overrides = {}

  date_obs = start_datetime
  for filename in _TESTDATA_TARGETS[target_name]:
    src_filename = os.path.join('testdata', 'input', filename)
    dest_filename = os.path.join(dest_dir, prefix + filename)
    shutil.copy(src_filename, dest_filename)
    print(f'### Copied filename {dest_filename}')
    fits = pyfits.open(dest_filename, mode='update', mmap=False)
    header = fits[0].header
    image = fits[0].data

    # Add a small amount of noise so it will have a unique hash.
    assert image.dtype == np.uint16
    if add_noise:
      image += np.random.randint(low=0,
                                 high=2,
                                 size=image.shape,
                                 dtype=image.dtype)

    content_hash = xxhash.xxh64(image.tobytes()).hexdigest()
    header['CONTHASH'] = content_hash

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
                          content_hashes=None,
                          dest_dir=None,
                          insert_metadata=True,
                          ra=100,
                          dec=45,
                          centers=None,
                          dither=0.01,
                          focal_length_m=0.35,
                          pixel_pitch_um=3.76,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=1080,
                          height_pixels=720,
                          temperature=-20,
                          exposure=300,
                          filter='H',
                          fill_value=None,
                          num_stars=100,
                          image_scale=1.0,
                          rotation_angle_deg=90.0):
  if not prefix:
    prefix = image_type + '_' + str(uuid.uuid4())

  number = len(ids) if ids else number

  ids = ids if ids else [f'{prefix}_{i:05}.fit' for i in range(number)]
  centers = centers if centers else [(ra, dec)] * number

  output_filenames = []

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

    if content_hashes is not None:
      content_hash = content_hashes[i]
    else:
      content_hash = xxhash.xxh64(image_data.tobytes()).hexdigest()

    center = _add_dither(centers[i], dither)
    date_obs = start_datetime + i * datetime.timedelta(seconds=exposure + 1)
    deg_per_pix = image_scale / 3600.0

    fits_header = {
        'SIMPLE': True,
        'BITPIX': 16,
        'CONTHASH': content_hash,
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
        'FOCALLEN': focal_length_m,
        'XPIXSZ': pixel_pitch_um,
        'YPIXSZ': pixel_pitch_um,
    }

    if image_type.lower() in ['light', 'flat']:
      fits_header['FILTER'] = filter

    header = normalize_header(fits_header)

    if image_type == 'Light':
      plate_solution = plate_solving.PlateSolution(
          image_filename=fits_filename,
          header=fits_header,
          num_stars=num_stars,
          center=center,
          image_scale=image_scale,
          rotation_angle=rotation_angle_deg)
    else:
      plate_solution = None

    if insert_metadata:
      key = ('image_metadata', content_hash)
      metadata = ImageMetadata(content_hash=content_hash,
                               filename=fits_filename,
                               header=header,
                               plate_solution=plate_solution)
      context.set(key, metadata, write_mode=WriteMode.DISK_ONLY)
      print(f'### Wrote key {key}')

    if not dest_dir:
      continue

    hdu = pyfits.PrimaryHDU(image_data)
    for k, v in fits_header.items():
      hdu.header[k] = v
    hdu.writeto(fits_filename)
    print(f'### Wrote FITS filename {fits_filename} with '
          f'content hash {content_hash}')
    output_filenames.append(fits_filename)

  return output_filenames
