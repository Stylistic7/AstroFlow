import subprocess
import sys
import tempfile
import os
from pathlib import Path
from multiprocessing import Pool
import astropy.wcs
import astropy.io.fits as pyfits
import warnings
from dataclasses import dataclass
import numpy as np
import traceback
from xisf import XISF

from cache import cache
import file_io


@dataclass
class PlateSolution:
  image_filename: str = None

  wcs_contents: bytes = None
  header: dict = None

  num_stars: int = 0

  center: list[float] = None
  rotation_angle: float = None
  footprint: list = None


# TODO: Don't open the WCS file multiple times.


def get_footprint(header):
  """Returns a 4x2 array. Order is clockwise starting with the bottom left corner."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    header['NAXIS1'] = header['IMAGEW']
    header['NAXIS2'] = header['IMAGEH']
    wcs = astropy.wcs.WCS(header=header)
    return wcs.calc_footprint(header=header).tolist()


def get_center(header):
  return header['CRVAL1'], header['CRVAL2']


def get_rotation_angle(header):
  return np.arctan2(header['CD1_2'], header['CD1_1'])


def get_num_stars(axy_file):
  with pyfits.open(axy_file) as fits:
    table = fits[1].data
    return table.shape[0]


def solve_field(filename):
  with tempfile.TemporaryDirectory() as temp_dir:
    if os.path.splitext(filename)[-1] in ['.xisf']:
      # TODO: Split this out into two round-trippable xisf_to_fits() and
      # fits_to_xisf() functions.
      xisf = XISF(filename)
      image = xisf.read_image(0).squeeze()
      img_meta = xisf.get_images_metadata()[0]
      fits_meta = img_meta['FITSKeywords']
      # print('### fits_meta', fits_meta)

      # TODO: Is this really necessary?
      image = image * 2**16
      image = image.astype(np.uint16)
      # print(f'image has min {np.min(image)} and max {np.max(image)}')
      # print(f'image has dtype {image.dtype}')

      hdu = pyfits.PrimaryHDU(image)
      for k, values in fits_meta.items():
        for v in values:
          hdu.header[k] = (v['value'], v['comment'])
          # print(f'### set FITS header {k} to {hdu.header[k]}')

      dest_filename = os.path.join(temp_dir, os.path.basename(filename))
      dest_filename = Path(dest_filename).with_suffix('.fits')

      dest_filename = os.path.join('/tmp', os.path.basename(dest_filename))

      print(f'Converting XISF to FITS {dest_filename}')
      hdu.writeto(dest_filename, overwrite=True)
    else:
      dest_filename = os.path.join(temp_dir, os.path.basename(filename))
      os.symlink(filename, dest_filename)

    try:
      # command = [
      #     "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
      #     "--no-background-subtraction", "--guess-scale", "--downsample", "64",
      #     "--crpix-center", "--depth", "100", dest_filename
      # ]
      command = [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--scale-low", "0.1", "--scale-high", "180.0", "--scale-units",
          "degwidth", "--guess-scale", "--downsample", "4", "--objs", "1000",
          "--tweak-order", "2", "--crpix-center",
          "--no-background-subtraction", dest_filename
      ]
      ret = subprocess.run(command, capture_output=True, timeout=15 * 60)
    except Exception as e:
      print(f'Could not plate solve file {dest_filename} due to exception')
      traceback.print_exc()
      return e

    wcs_file = Path(dest_filename).with_suffix('.wcs')
    axy_file = Path(dest_filename).with_suffix('.axy')
    if os.path.exists(wcs_file) and os.path.exists(axy_file):
      wcs_contents = open(wcs_file, 'rb').read()
      with pyfits.open(wcs_file) as fits:
        header = fits[0].header
        print(f'### Sucessfully plate-solved file {dest_filename}')
        return PlateSolution(image_filename=filename,
                             wcs_contents=wcs_contents,
                             header=header,
                             center=get_center(header),
                             footprint=get_footprint(header),
                             rotation_angle=get_rotation_angle(header),
                             num_stars=get_num_stars(axy_file))
    else:
      # TODO: Emit warnings through Metaflow's reporting system.
      print(f'Could not plate-solve file {filename}')
      return None


def solve_fields(filenames):
  with Pool() as p:
    solutions = p.star(solve_field, filenames)
    return {f: s for f, s in zip(filenames, solutions) if s != None}


if __name__ == '__main__':
  soln = solve_field(sys.argv[1])
  print('Got plate solution', soln)
