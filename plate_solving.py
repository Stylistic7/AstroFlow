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

from cache import cache


@dataclass
class PlateSolution:
  image_filename: str = None

  wcs_contents: bytes = None

  num_stars: int = 0

  center: list[float] = None
  rotation_angle: float = None
  footprint: list = None


def get_footprint(wcs_file):
  """Returns a 4x2 array. Order is clockwise starting with the bottom left corner."""
  with pyfits.open(wcs_file) as fits:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      header = fits[0].header
      header['NAXIS1'] = header['IMAGEW']
      header['NAXIS2'] = header['IMAGEH']
      wcs = astropy.wcs.WCS(header=header, fobj=fits)
      return wcs.calc_footprint(header=header).tolist()


def get_center(wcs_file):
  with pyfits.open(wcs_file) as fits:
    header = fits[0].header
    return header['CRVAL1'], header['CRVAL2']


def get_rotation_angle(wcs_file):
  with pyfits.open(wcs_file) as fits:
    header = fits[0].header
    return np.arctan2(header['CD1_2'], header['CD1_1'])


def get_num_stars(axy_file):
  with pyfits.open(axy_file) as fits:
    table = fits[1].data
    return table.shape[0]


@cache.memoize()
def solve_field(filename):
  with tempfile.TemporaryDirectory() as temp_dir:
    dest_filename = os.path.join(temp_dir, os.path.basename(filename))
    os.symlink(filename, dest_filename)

    try:
      command = [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--no-background-subtraction", "--guess-scale", "--downsample", "4",
          "--objs", "1000", dest_filename
      ]
      ret = subprocess.run(command, capture_output=True, timeout=120)
    except:
      return None

    wcs_file = Path(dest_filename).with_suffix('.wcs')
    axy_file = Path(dest_filename).with_suffix('.axy')
    if os.path.exists(wcs_file) and os.path.exists(axy_file):
      return PlateSolution(image_filename=filename,
                           wcs_contents=open(wcs_file, 'rb').read(),
                           center=get_center(wcs_file),
                           footprint=get_footprint(wcs_file),
                           rotation_angle=get_rotation_angle(wcs_file),
                           num_stars=get_num_stars(axy_file))
    else:
      # TODO: Emit warnings through Metaflow's reporting system.
      print(f'Could not plate-solve file {filename}')
      return None


def solve_fields(filenames):
  with Pool() as p:
    solutions = p.map(solve_field, filenames)
    return {f: s for f, s in zip(filenames, solutions) if s != None}


if __name__ == '__main__':
  wcs, stdout = solve_field(sys.argv[1])
