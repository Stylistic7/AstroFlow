import subprocess
import sys
import tempfile
import os
from pathlib import Path
import astropy.wcs  # type: ignore
import astropy.io.fits as pyfits  # type: ignore
import warnings
from dataclasses import dataclass
import numpy as np
import traceback
from xisf import XISF  # type: ignore
import time
from typing import Any

import throughput


@dataclass
class PlateSolution:
  image_filename: str | None = None
  header: dict | None = None
  num_stars: int | None = None
  center: tuple[float, float] | None = None
  image_scale: float | None = None
  rotation_angle: float | None = None
  footprint: list | None = None


def get_footprint(header: dict[str, Any]) -> list[float]:
  """Returns a 4x2 array. Order is clockwise starting with the bottom left corner."""
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    header['NAXIS1'] = header['IMAGEW']
    header['NAXIS2'] = header['IMAGEH']
    wcs = astropy.wcs.WCS(header=header)
    return wcs.calc_footprint(header=header).tolist()


def get_center(header: dict[str, Any]) -> tuple[float, float]:
  return float(header['CRVAL1']), float(header['CRVAL2'])


def get_rotation_angle(header: dict[str, Any]) -> float:
  return np.arctan2(header['CD1_2'], header['CD1_1'])


def get_image_scale(header: dict[str, Any]) -> float:
  scale_x = np.linalg.norm(np.array([header['CD1_1'], header['CD2_1']]))
  scale_y = np.linalg.norm(np.array([header['CD1_2'], header['CD2_2']]))
  assert np.isclose(
      scale_x, scale_y, atol=0.05 / 3600,
      rtol=0), f'scale_x {scale_x} is not close to scale_y {scale_y}'

  # The result is in arcseconds per pixel.
  image_scale = scale_x * 3600.0
  return float(image_scale)


def get_num_stars(axy_file: Path | str) -> int:
  fits = pyfits.open(axy_file)
  table = fits[1].data
  fits.close()
  return table.shape[0]


def simplify_header(header: dict[str, Any]) -> dict[str, Any]:
  output_header = {}
  for k in header.keys():
    if 'COMMENT' in k or 'HISTORY' in k:
      continue
    output_header[k] = header[k]

  return output_header


def _solve_field(filename: Path | str,
                 arguments: list[str]) -> PlateSolution | Exception | None:
  # TODO: Only copy the file to a temp dir once.
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

      dest_filename = Path(os.path.join(
          temp_dir, os.path.basename(filename))).with_suffix('.fits')

      hdu.writeto(dest_filename, overwrite=True)
    else:
      dest_filename = Path(os.path.join(temp_dir, os.path.basename(filename)))
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
      fits = pyfits.open(wcs_file)
      header = simplify_header(fits[0].header)
      try:
        num_stars = get_num_stars(axy_file)
        soln = PlateSolution(image_filename=str(filename),
                             header=header,
                             center=get_center(header),
                             image_scale=get_image_scale(header),
                             footprint=get_footprint(header),
                             rotation_angle=get_rotation_angle(header),
                             num_stars=num_stars)
        print(f'### Sucessfully plate-solved file {filename}, '
              f'has {num_stars} stars')
        return soln
      except Exception as e:
        print(f'Could not plate solve file {filename} due to exception')
        traceback.print_exc()
        return e
      finally:
        fits.close()
    else:
      # TODO: Emit warnings through Metaflow's reporting system.
      print(f'Could not plate-solve file {filename}')
      return None


# TODO: Make a "junk drawer" video that shows everything that can't be plate-
# solved. The user can look at it and decide whether to improve the solver's
# settings, or to throw the images away.
def solve_field(filename):
  assert filename

  # TODO: These options must be exposed to the user.
  trial_arguments = [
      [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--scale-low", "1.0", "--scale-high", "10.0", "--scale-units",
          "degwidth", "--guess-scale", "--downsample", "16", "--tweak-order",
          "4", "--crpix-center", "--no-background-subtraction", "--depth", "10",
          "--cpulimit", "5"
      ],
      [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--scale-low", "0.5", "--scale-high", "10.0", "--scale-units",
          "degwidth", "--guess-scale", "--downsample", "16", "--tweak-order",
          "4", "--crpix-center", "--no-background-subtraction", "--depth",
          "100", "--cpulimit", "60"
      ],
      [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--downsample", "16", "--tweak-order", "4", "--crpix-center",
          "--no-background-subtraction", "--cpulimit", "120"
      ],
      [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--downsample", "16", "--tweak-order", "4", "--crpix-center",
          "--cpulimit", "120"
      ],
      [
          "solve-field", "--overwrite", "--no-plots", "--new-fits", "none",
          "--scale-low", "0.5", "--scale-high", "10.0", "--scale-units",
          "degwidth", "--guess-scale", "--downsample", "4", "--tweak-order",
          "4", "--crpix-center", "--no-background-subtraction", "--depth",
          "500", "--cpulimit", "300"
      ]
  ]

  start_time = time.time()
  for i, arguments in enumerate(trial_arguments):
    result = _solve_field(filename, arguments)
    if type(result) is PlateSolution:
      end_time = time.time()
      throughput.add_sample('plate_solve', end_time - start_time)
      return result

  print(f'Could not plate-solve file {filename}')
  return None


if __name__ == '__main__':
  soln = solve_field(sys.argv[1])
  print(soln.footprint)
