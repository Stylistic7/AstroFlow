import os

import file_io
import subframe_selector
import diskcache

cache = diskcache.Cache(os.path.join(os.getcwd(), '.cache'))


def run(light_frames, image_scale):
  if 'measurements' not in cache:
    cache['measurements'] = {}

  # Add mesurements for new files.
  added_files = set(light_frames) - set(cache['measurements'].keys())
  if len(added_files):
    cache['measurements'] |= subframe_selector.measure_frames(
        added_files, image_scale)

  return {
      k: cache['measurements'][k]
      for k in light_frames if k in cache['measurements']
  }