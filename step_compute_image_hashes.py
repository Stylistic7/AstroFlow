from multiprocessing import Pool
import os

import file_io


def _compute_hashes(filenames):
  with Pool() as p:
    hashes = p.map(file_io.hash_of_image_data, filenames)
    return {f: h for f, h in zip(filenames, hashes)}


def run(image_hashes, light_frames):
  # Remove files that no longer exist in the filesystem.
  image_hashes = {k: v for k, v in image_hashes.items() if os.path.exists(k)}

  # Add hashes for new files.
  added_files = set(light_frames) - set(image_hashes.keys())
  if len(added_files):
    print(f'Computing hashes for {len(added_files)} new images...')
    image_hashes |= _compute_hashes(added_files)

  return image_hashes