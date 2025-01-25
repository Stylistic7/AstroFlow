import numpy as np
from pathlib import Path
import xxhash

from file_io.fits_file_io import read_fits_header, read_fits_image
from file_io.xisf_file_io import read_xisf_header, read_xisf_image

_CONTENT_HASH_KEY = 'CONTHASH'


def read_image(filename: Path | str) -> np.ndarray:
  path = Path(filename)
  if path.suffix.lower() in ['.xisf']:
    return read_xisf_image(path)
  elif path.suffix.lower() in ['.fit', '.fits']:
    return read_fits_image(path)

  assert False, f'Unknown file type {filename}'


def read_header(filename: Path | str) -> dict:
  path = Path(filename)
  if path.suffix.lower() in ['.xisf']:
    return read_xisf_header(path)
  elif path.suffix.lower() in ['.fit', '.fits']:
    return read_fits_header(path)

  assert False, f'Unknown file type {filename}'


def hash_of_image_data(filename: Path | str) -> str:
  return xxhash.xxh64(read_image(filename).tobytes()).hexdigest()


# TODO: Strongly-type content hashes.
def get_content_hash(filename: Path | str) -> str:
  header = read_header(filename)
  if _CONTENT_HASH_KEY in header:
    return header[_CONTENT_HASH_KEY]
  else:
    return hash_of_image_data(filename)
