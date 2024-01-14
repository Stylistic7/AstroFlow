import os
import time
import shutil
import concurrent.futures

import file_io
import plate_solving
import throughput
import round_robin

import ray

from astro_flow_direct import global_registry


@ray.remote
def hash_image(filename):
  return file_io.hash_of_image_data(filename)


# TODO: Need to properly handle renamed or moved or deleted files.
def compute_image_hashes(all_frames, image_hashes, quarantine_dir=None):
  # Set the filename to None for any files that no longer exist. Their hashes
  # remain, though, and they can be found again with some user input.
  prev_num_missing_files = sum(1 for f in image_hashes.values() if not f)

  def exists_or_none(f):
    if f and os.path.exists(f):
      return f
    else:
      return None

  image_hashes = {h: exists_or_none(f) for h, f in image_hashes.items()}
  num_missing_files = sum(1 for f in image_hashes.values() if not f)
  print(f'### {num_missing_files - prev_num_missing_files} '
        f'files have gone missing.')

  new_filenames = list(set(all_frames) - set(image_hashes.values()))

  print(f'### Computing image hashes for {len(new_filenames)} new files...')
  ref_to_filename = {hash_image.remote(f): f for f in new_filenames}
  remaining_refs = list(ref_to_filename.keys())
  while len(remaining_refs):
    ready_refs, remaining_refs = ray.wait(remaining_refs)
    for ref in ready_refs:
      filename = ref_to_filename[ref]
      result = ray.get(ref)

      if result in image_hashes and image_hashes[result] is not None:
        if quarantine_dir:
          print(f'### Quarantining file {filename} because it is a duplicate')
          shutil.move(filename,
                      os.path.join(quarantine_dir, os.path.basename(filename)))
        continue

      image_hashes[result] = filename

  return image_hashes


# TODO: This will scale poorly when there are hundreds of thousands of images.
def read_fits_headers(image_hashes, fits_headers):
  new_image_hashes = list(set(image_hashes.keys()) - set(fits_headers.keys()))
  new_filenames = [
      f for h, f in image_hashes.items() if f and h in new_image_hashes and
      os.path.splitext(f)[-1] in ['.fit', '.fits']
  ]
  assert None not in new_filenames

  print(f'### Reading FITS headers for {len(new_filenames)} files...')
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_filename = {
        executor.submit(file_io.read_fits, filename): filename
        for filename in new_filenames
    }

    for future in concurrent.futures.as_completed(future_to_filename):
      filename = future_to_filename[future]
      result = future.result()
      hash = next(h for h, f in image_hashes.items() if f == filename)
      assert hash
      fits_headers[hash] = result

  return fits_headers


def plate_solve(images,
                image_hashes,
                plate_solutions,
                quarantine_dir=None,
                callback_number=None,
                callback=None):
  new_image_hashes = list(set(images) - set(plate_solutions.keys()))
  new_filenames = [
      f for h, f in image_hashes.items() if f and h in new_image_hashes
  ]
  assert None not in new_filenames
  print(f'### Plate solving {len(new_filenames)} newly-added files...')

  new_filenames = round_robin.round_robin_by_directory(new_filenames,
                                                       cluster=5)

  future_to_filename = {
      plate_solving.solve_field.remote(filename): filename
      for filename in new_filenames
  }

  num_solved = 0
  for future in future_to_filename.keys():
    filename = future_to_filename[future]
    result = ray.get(future)

    if not result:
      if quarantine_dir:
        print(f'### Quarantining file {filename} because it could not '
              f'be plate-solved')
        shutil.move(filename,
                    os.path.join(quarantine_dir, os.path.basename(filename)))

    hash = next((h for h, f in image_hashes.items() if f == filename), None)
    assert hash, f"Could not find filename {f} in image_hashes"

    num_solved += 1
    plate_solutions[hash] = result

    if callback:
      assert callback_number

    if callback and num_solved and num_solved % callback_number == 0:
      callback(plate_solutions)

  if callback:
    callback(plate_solutions)

  return plate_solutions