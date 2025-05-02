from ray_infra import ray_local

from pathlib import Path
import tempfile
import datetime

from context import context


def make_test_context(
) -> tuple[tempfile.TemporaryDirectory, tempfile.TemporaryDirectory]:
  context.clear()

  context.quarantined_files = []

  context.mock_pixinsight = True

  input_dir_obj = tempfile.TemporaryDirectory()
  context.input_dirs = [Path(input_dir_obj.name)]

  output_dir_obj = tempfile.TemporaryDirectory()
  context.output_dir = Path(output_dir_obj.name)

  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(hours=8)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3
  context.subframe_thumbnail_width = 1200
  context.master_thumbnail_width = 2160
  context.subframe_mosaic_samples = 500

  return input_dir_obj, output_dir_obj
