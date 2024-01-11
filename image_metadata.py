import os
import time
import shutil
import concurrent.futures

import file_io
import plate_solving
import throughput
import round_robin

import ray

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
  print('some new filenames', new_filenames[0:10])

  print(f'### Computing image hashes for {len(new_filenames)} files...')
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_filename = {
        executor.submit(file_io.hash_of_image_data, filename): filename
        for filename in new_filenames
    }

    for future in concurrent.futures.as_completed(future_to_filename):
      filename = future_to_filename[future]
      print(f'future for filename {filename} is done')
      result = future.result()
      print(f'result it {result}')

      # TODO: Report these failures to the user in the "library health" panel.
      if not result:
        print(f'### Got falsy result {result} for filename {filename}')

      # TODO: Do something wiser than just asserting if there are duplicate
      # images.
      if result in image_hashes and image_hashes[result] is not None:
        if quarantine_dir:
          print(f'### Quarantining file {filename} because it is a duplicate')
          shutil.move(filename,
                      os.path.join(quarantine_dir, os.path.basename(filename)))
        continue

      print(f'### image_hash[{result}] = {filename}')
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


# TODO: This will scale poorly when there are hundreds of thousands of images.
#
# TODO: Make this iterative (progressive). Find runs in light frames. Iterate
# across the runs, solving the first image from each run, then the second image
# from each run, etc. This will allow the user to see what's happening, without
# having to wait to solve the entire library.
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

  num_solved = 0
  # with concurrent.futures.ThreadPoolExecutor() as executor:

  future_to_filename = {
      # executor.submit(plate_solving.solve_field, filename): filename
      plate_solving.solve_field.remote(filename): filename
      for filename in new_filenames
  }

  # for future in concurrent.futures.as_completed(future_to_filename):
  for future in future_to_filename.keys():
    filename = future_to_filename[future]
    # result = future.result()
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