import os

import autorun_log_analysis


def run(light_frames, bad_file_dir=None):
  filtered_light_frames = autorun_log_analysis.remove_frames_during_settling(
      light_frames)
  num_removed = len(light_frames) - len(filtered_light_frames)
  if num_removed:
    print(f'Removed {num_removed} '
          f'light frames during settling periods.')

  # Quarantine files that couldn't be plate solved.
  if bad_file_dir:
    bad_files = set(light_frames) - set(filtered_light_frames)
    if len(bad_files):
      print(f'Quarantining {len(bad_files)} bad files')
      os.makedirs(bad_file_dir, exist_ok=True)
      for filename in bad_files:
        os.rename(filename,
                  os.path.join(bad_file_dir, os.path.basename(filename)))

  return filtered_light_frames