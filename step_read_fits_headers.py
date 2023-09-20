from multiprocessing import Pool
import os

import file_io


def _read_fits_headers(filenames):
  with Pool() as p:
    headers = p.map(file_io.read_fits, filenames)
    return {f: h for f, h in zip(filenames, headers)}


def run(headers, light_frames):
  # Remove files that no longer exist in the filesystem.
  headers = {k: v for k, v in headers.items() if os.path.exists(k)}

  # Add hashes for new files.
  added_files = set(light_frames) - set(headers.keys())
  if len(added_files):
    print(f'Computing hashes for {len(added_files)} new images...')
    headers |= _read_fits_headers(added_files)

  return headers