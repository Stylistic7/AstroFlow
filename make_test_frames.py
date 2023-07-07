import os
import astropy.io.fits as pyfits
import numpy as np
import uuid


def make_light_frames(project_dir, ra, dec, dither):
  random_id = str(uuid.uuid4())
  fits_filenames = []
  for i in range(10):
    fits_filename = os.path.join(project_dir, f'Light_{random_id}_{i:05}.fit')
    fits_filenames.append(fits_filename)

    image_data = np.arange(100.0)
    hdu = pyfits.PrimaryHDU(image_data)

    hdu.header['EXPOSURE'] = 300.0

    hdu.header['CRVAL1'] = np.random.normal(loc=ra, scale=dither)
    hdu.header['CRVAL2'] = np.random.normal(loc=dec, scale=dither)

    hdu.writeto(fits_filename)

  return fits_filenames