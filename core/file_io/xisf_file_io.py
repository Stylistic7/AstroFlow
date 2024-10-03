import numpy as np
from pathlib import Path
from xisf import XISF  # type: ignore
from typing import Any

from file_io.normalize_header import normalize_header


def read_xisf_header(filename: Path | str, debug: bool = False) -> dict:
  xisf = XISF(filename)

  ims_meta = xisf.get_images_metadata()
  assert len(ims_meta) == 1

  def _eval(value: Any) -> Any:
    try:
      return eval(value)
    except:
      return value

  fits_header = ims_meta[0]['FITSKeywords']

  if debug:
    for k, v in fits_header.items():
      print(f'Raw FITS entry {k} = {v}')

  header = {k: _eval(fits_header[k][0]['value']) for k in fits_header.keys()}

  width, height, depth = ims_meta[0]['geometry']
  assert depth == 1
  header['NAXIS1'], header['NAXIS2'] = width, height
  header['NAXIS'] = 2

  header = normalize_header(header)

  if debug:
    for k, v in header.items():
      print(f'Normalized FITS entry {k} = {repr(v)}')

  return header


def read_xisf_image(filename: Path | str) -> np.ndarray:
  xisf = XISF(str(filename))
  return xisf.read_image(0).squeeze().astype(np.float32)


def add_xisf_metadata(filename: Path | str, dest_filename: Path | str,
                      metadata: dict[str, Any]) -> None:
  """Adds the given metadata to the file.
  
  filename and dest_filename can be the same.
  """
  xisf = XISF(filename)
  image = xisf.read_image(0)
  file_meta = xisf.get_file_metadata()
  ims_meta = xisf.get_images_metadata()

  fits_header = ims_meta[0]['FITSKeywords']
  for k, v in metadata.items():
    fits_header[k] = [{'value': str(v), 'comment': ''}]

  XISF.write(dest_filename,
             image,
             creator_app="AstroFlow",
             image_metadata=ims_meta[0],
             xisf_metadata=file_meta)
