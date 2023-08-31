import subprocess
import sys
import tempfile
import os
from pathlib import Path
from multiprocessing import Pool
import astropy.wcs
import astropy.io.fits as pyfits
import warnings
import diskcache

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))


def calc_footprint(wcs_file):
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
      return None, None, None

    wcs_file = Path(dest_filename).with_suffix('.wcs')
    if os.path.exists(wcs_file):
      wcs = open(wcs_file, 'rb').read()
      center = get_center(wcs_file)
      footprint = calc_footprint(wcs_file)

      return center, footprint, wcs
    else:
      # TODO: Emit warnings through Metaflow's reporting system.
      print(f'Could not plate-solve file {filename}')
      return None, None, None


def solve_fields(filenames):
  with Pool() as p:
    solutions = p.map(solve_field, filenames)
    return {f: s for f, s in zip(filenames, solutions) if s[0] != None}


if __name__ == '__main__':
  wcs, stdout = solve_field(sys.argv[1])
