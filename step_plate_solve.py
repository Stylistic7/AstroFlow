import os

import plate_solving


def run(plate_solutions, light_frames, bad_file_dir=None):
  # Remove files that no longer exist in the filesystem.
  num_files = len(plate_solutions)
  plate_solutions = {
      k: v
      for k, v in plate_solutions.items() if os.path.exists(k)
  }
  num_removed = len(plate_solutions) - num_files
  if num_removed:
    print(f'Removed {num_removed} deleted images')

  # Add hashes for new files.
  added_files = set(light_frames) - set(plate_solutions.keys())
  if len(added_files):
    print(f'Plate solving {len(added_files)} new images...')
    plate_solutions |= plate_solving.solve_fields(added_files)

  # Quarantine files that couldn't be plate solved.
  if bad_file_dir:
    bad_files = set(light_frames) - set(plate_solutions.keys())
    if len(bad_files):
      print(f'Quarantining {len(bad_files)} bad files')
      os.makedirs(bad_file_dir, exist_ok=True)
      for filename in bad_files:
        os.rename(filename,
                  os.path.join(bad_file_dir, os.path.basename(filename)))

  return plate_solutions