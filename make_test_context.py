import ray_local

import ray
import tempfile
import datetime

from cache import cache
import ray_diskcache
from context import Context


def make_test_context():
  ray_diskcache.clear()
  cache.clear()

  context = Context()

  context.image_hashes = ray.put({})
  context.fits_headers = ray.put({})
  context.plate_solutions = ray.put({})
  context.quarantined_files = ray.put([])

  context.stacks = ray.put({})
  context.targets = ray.put({})
  context.mosaics = ray.put({})

  input_dir_obj = tempfile.TemporaryDirectory()
  context.input_dirs.append(input_dir_obj.name)

  output_dir_obj = tempfile.TemporaryDirectory()
  context.output_dir = output_dir_obj.name

  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160

  return context, input_dir_obj, output_dir_obj
