import ray_local

import ray
import tempfile
import datetime

from cache import cache
import ray_diskcache
from context import Context
from new_context import new_context


def make_test_context():
  ray_diskcache.clear()
  cache.clear()
  new_context.clear()

  new_context.image_hashes = {}
  new_context.fits_headers = {}
  new_context.plate_solutions = {}
  new_context.quarantined_files = []

  new_context.stacks = {}
  new_context.targets = {}
  new_context.mosaics = {}

  new_context.regroup = False

  input_dir_obj = tempfile.TemporaryDirectory()
  new_context.input_dirs = [input_dir_obj.name]

  output_dir_obj = tempfile.TemporaryDirectory()
  new_context.output_dir = output_dir_obj.name

  new_context.cosine_similarity_threshold = 0.99999
  new_context.min_component_members = 3
  new_context.max_run_gap = datetime.timedelta(hours=8)
  new_context.min_frames_in_run = 3
  new_context.min_frames_in_stack = 3
  new_context.subframe_thumbnail_width = 1200
  new_context.master_thumbnail_width = 2160

  new_context.mock_stacking = True

  return input_dir_obj, output_dir_obj
