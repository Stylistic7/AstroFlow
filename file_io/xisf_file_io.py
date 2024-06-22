import numpy as np
from pathlib import Path
from xisf import XISF  # type: ignore

from file_io.normalize_header import normalize_header


def read_xisf_header(filename: Path | str, debug: bool = False) -> dict:
  xisf = XISF(filename)

  ims_meta = xisf.get_images_metadata()
  assert len(ims_meta) == 1

  def _eval(value):
    try:
      return eval(value)
    except:
      return value

  fits_header = ims_meta[0]['FITSKeywords']
  header = {k: _eval(fits_header[k][0]['value']) for k in fits_header.keys()}

  width, height, depth = ims_meta[0]['geometry']
  assert depth == 1
  header['NAXIS1'], header['NAXIS2'] = width, height
  header['NAXIS'] = 2

  header = normalize_header(header)

  if debug:
    for k, v in header.items():
      print(k, '=', repr(v))

  return header


def read_xisf_image(filename: Path | str) -> np.ndarray:
  xisf = XISF(str(filename))
  return xisf.read_image(0).squeeze()
