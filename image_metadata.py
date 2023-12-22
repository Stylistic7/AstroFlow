import os
import time
import shutil
import concurrent.futures

import file_io
import plate_solving
import throughput


# TODO: Need to properly handle renamed or moved or deleted files.
def compute_image_hashes(all_frames, image_hashes, quarantine_dir=None):
  # Set the filename to None for any files that no longer exist. Their hashes
  # remain, though, and they can be found again with some user input.
  prev_num_missing_files = sum(1 for f in image_hashes.values() if not f)
  image_hashes = {
      h: f if os.path.exists(f) else None
      for h, f in image_hashes.items()
  }
  num_missing_files = sum(1 for f in image_hashes.values() if not f)
  print(f'### {num_missing_files - prev_num_missing_files} '
        f'files have gone missing.')

  new_filenames = list(set(all_frames) - set(image_hashes.values()))

  print(f'### Computing image hashes for {len(new_filenames)} files...')
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_filename = {
        executor.submit(file_io.hash_of_image_data, filename): filename
        for filename in new_filenames
    }

    for future in concurrent.futures.as_completed(future_to_filename):
      filename = future_to_filename[future]
      result = future.result()

      # TODO: Report these failures to the user in the "library health" panel.
      if not result:
        print(f'### Got falsy result {result} for filename {filename}')

      # TODO: Do something wiser than just asserting if there are duplicate
      # images.
      if result in image_hashes:
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
      f for h, f in image_hashes.items() if h in new_image_hashes and
      os.path.splitext(f)[-1] in ['.fit', '.fits']
  ]

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
def plate_solve(images, image_hashes, plate_solutions, quarantine_dir=None):
  new_image_hashes = list(set(images) - set(plate_solutions.keys()))
  new_filenames = [f for h, f in image_hashes.items() if h in new_image_hashes]

  print(f'### Plate solving {len(new_filenames)} newly-added files...')
  num_remaining = len(new_filenames)
  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_filename = {
        executor.submit(plate_solving.solve_field, filename): filename
        for filename in new_filenames
    }

    for future in concurrent.futures.as_completed(future_to_filename):
      filename = future_to_filename[future]
      result = future.result()

      if not result:
        if quarantine_dir:
          print(f'### Quarantining file {filename} because it could not '
                f'be plate-solved')
          shutil.move(filename,
                      os.path.join(quarantine_dir, os.path.basename(filename)))
        continue

      hash = next(h for h, f in image_hashes.items() if f == filename)
      plate_solutions[hash] = result
      num_remaining -= 1
      eta = time.strftime(
          "%H:%M:%S",
          time.gmtime(throughput.estimate_time('plate_solve', num_remaining)))
      print(f'### ETA {eta}')

  return plate_solutions