import os

import file_io
import subframe_selector


def run(measurements, image_scale, light_frames):
  # Remove files that no longer exist in the filesystem.
  measurements = {k: v for k, v in measurements.items() if os.path.exists(k)}

  # Add mesurements for new files.
  added_files = set(light_frames) - set(measurements.keys())
  if len(added_files):
    measurements |= subframe_selector.measure_frames(added_files, image_scale)

  return measurements