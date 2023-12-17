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
import objsize
import concurrent.futures
import copy

from cache import cache
import file_io


@dataclass
class PlateSolution:
  image_filename: str = None
  header: dict = None
  num_stars: int = 0
  center: list[float] = None
  image_scale: float = None
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


def get_image_scale(header):
  scale_x = np.linalg.norm(np.array([header['CD1_1'], header['CD2_1']]))
  scale_y = np.linalg.norm(np.array([header['CD1_2'], header['CD2_2']]))
  assert np.isclose(
      scale_x, scale_y, atol=0.01 / 3600,
      rtol=0), f'scale_x {scale_x} is not close to scale_y {scale_y}'

  # The result is in arcseconds per pixel.
  image_scale = scale_x * 3600.0
  return image_scale


def get_num_stars(axy_file):
  with pyfits.open(axy_file) as fits:
    table = fits[1].data
    return table.shape[0]


def simplify_header(header):
  output_header = {}
  for k in header.keys():
    if 'COMMENT' in k or 'HISTORY' in k: continue
    output_header[k] = header[k]

  return output_header


def _solve_field(filename, arguments):
  with tempfile.TemporaryDirectory() as temp_dir:
    if os.path.splitext(filename)[-1] in ['.xisf']:
      # TODO: Split this out into two round-trippable xisf_to_fits() and
      # fits_to_xisf() functions.
      xisf = XISF(filename)
      image = xisf.read_image(0).squeeze()
      img_meta = xisf.get_images_metadata()[0]
      fits_meta = img_meta['FITSKeywords']

      hdu = pyfits.PrimaryHDU(image)
      for k, values in fits_meta.items():
        for v in values:
          hdu.header[k] = (v['value'], v['comment'])

      dest_filename = os.path.join(temp_dir, os.path.basename(filename))
      dest_filename = Path(dest_filename).with_suffix('.fits')

      # print(f'Converting XISF to FITS {dest_filename}')
      hdu.writeto(dest_filename, overwrite=True)
    else:
      dest_filename = os.path.join(temp_dir, os.path.basename(filename))
      os.symlink(filename, dest_filename)

    try:
      command = arguments + [dest_filename]
      ret = subprocess.run(command, capture_output=True)
    except Exception as e:
      # print(f'Could not plate solve file {dest_filename} due to exception')
      # traceback.print_exc()
      return e

    wcs_file = Path(dest_filename).with_suffix('.wcs')
    axy_file = Path(dest_filename).with_suffix('.axy')
    if os.path.exists(wcs_file) and os.path.exists(axy_file):
      wcs_contents = open(wcs_file, 'rb').read()
      with pyfits.open(wcs_file) as fits:
        header = simplify_header(fits[0].header)
        try:
          soln = PlateSolution(image_filename=filename,
                               header=header,
                               center=get_center(header),
                               image_scale=get_image_scale(header),
                               footprint=get_footprint(header),
                               rotation_angle=get_rotation_angle(header),
                               num_stars=get_num_stars(axy_file))
          # print(f'### Sucessfully plate-solved file {dest_filename}')
          return soln
        except Exception as e:
          # print(f'Could not plate solve file {dest_filename} due to exception')
          # traceback.print_exc()
          return e
    else:
      # TODO: Emit warnings through Metaflow's reporting system.
      # print(f'Could not plate-solve file {filename}')
      return None


# def solve_field(filename):
#   for t in trial_arguments:
#     result = _solve_field(filename, arguments=t)
#     if result:
#       return result


def solve_fields(filenames):
  print('### Entering solve_fields')

  # TODO: These options must be exposed to the user.
  trial_arguments = [[
      "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
      "--scale-low", "1", "--scale-high", "10.0", "--scale-units", "degwidth",
      "--guess-scale", "--downsample", "4", "--objs", "100", "--tweak-order",
      "4", "--crpix-center", "--no-background-subtraction", "--cpulimit", "120"
  ],
                     [
                         "solve-field", "--overwrite", "--no-plots",
                         "--new-fits", "none", "--scale-low", "1",
                         "--scale-high", "10.0", "--scale-units", "degwidth",
                         "--guess-scale", "--downsample", "4", "--objs", "100",
                         "--tweak-order", "4", "--crpix-center", "--cpulimit",
                         "120"
                     ]]

  solutions = {}

  with concurrent.futures.ThreadPoolExecutor() as executor:
    remaining_filenames = copy.copy(filenames)
    for t, arguments in enumerate(trial_arguments):
      print(f'### Using trial_arguments[{t}]')
      future_to_filename = {
          executor.submit(_solve_field, filename, arguments): filename
          for filename in filenames
      }

      num_solved = 0
      for i, future in enumerate(
          concurrent.futures.as_completed(future_to_filename)):
        filename = future_to_filename[future]
        result = future.result()
        if not result: continue
        num_solved += 1
        solutions[filename] = result
        remaining_filenames.remove(filename)

        if i > 0 and i % 100 == 0:
          print(f'### Solved {num_solved} / {i} images')

      if not remaining_filenames:
        return solutions

  print(f'### Failed to solve {len(filenames) - len(solutions)} images')

  return solutions

  # all_solutions = {}
  # with Pool() as p:
  #   for chunk in chunker(filenames, 100):
  #     solutions = p.star(solve_field, chunk)
  #     all_solutions |= {f: s for f, s in zip(chunk, solutions) if s != None}

  # return all_solutions


if __name__ == '__main__':
  soln = solve_field(sys.argv[1])
  print(soln.footprint)
