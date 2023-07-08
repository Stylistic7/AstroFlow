# Algorithm idea

# Extract all the dates from the light frames.

# For each date:
# Look for flat frames on the same dates.

import file_io


def associate_flat_frames(light_frames, flat_frames):
  light_frame_dates = set(file_io.read_fits(f)['DATE'] for f in light_frames)
  flat_frame_dates = set(file_io.read_fits(f)['DATE'] for f in light_frames)

  print('light_frame_dates', light_frame_dates)
  print('flat_frame_dates', flat_frame_dates)