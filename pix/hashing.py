import os
import xxhash

from .file_io import read_xisf_header, hash_of_xisf_image_data

_CONTENT_HASH_KEY = 'XISF:AutoPix:ContentHash'


# TODO: Maybe unify these.
def hash_string(x):
  return xxhash.xxh64(x).hexdigest()


def hash_list(x):
  return xxhash.xxh64(str(sorted(set(iter(x))))).hexdigest()


def get_content_hash(filename):
  """Returns the file's explicit content hash, or the hash of its image data."""
  assert os.path.exists(filename)
  return read_xisf_header(filename).get(_CONTENT_HASH_KEY,
                                        hash_of_xisf_image_data(filename))
