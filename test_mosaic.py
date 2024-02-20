import ray_local

import unittest
import tempfile
import datetime
import ray

import file_io
from cache import cache
from context import Context
from make_test_frames import copy_testdata_frames
import ray_diskcache
from image_metadata import scatter_compute_image_metadata
from spatial_association import find_components_by_cosine_similarity
from stack_identification import identify_stacks
from calibration_frames import associate_calibration_frames
from target_identification import group_stacks_into_targets
from mosaics.identification import identify_mosiacs
from step_mosaic import scatter_merge_mosaics


@ray.remote
def _compute_graph_fragment(context, new_filenames):
  context = ray.get(
      scatter_compute_image_metadata.remote(context, new_filenames))
  context = ray.get(find_components_by_cosine_similarity.remote(context))
  context = ray.get(identify_stacks.remote(context))
  context = ray.get(associate_calibration_frames.remote(context))
  context = ray.get(group_stacks_into_targets.remote(context))
  context = ray.get(identify_mosiacs.remote(context))
  context = ray.get(scatter_merge_mosaics.remote(context, preview=True))

  return context


class TestMosaic(unittest.TestCase):

  def setUp(self):
    ray_diskcache.clear()
    cache.clear()

    # Start with an empty context.
    self.context = Context()
    self.context.image_hashes = ray.put({})
    self.context.fits_headers = ray.put({})
    self.context.plate_solutions = ray.put({})

    self.input_dir = tempfile.TemporaryDirectory()
    self.context.input_dirs.append(self.input_dir.name)

    self.output_dir = tempfile.TemporaryDirectory()
    self.context.output_dir = self.output_dir.name

  def test_orion_mosaic(self):
    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(target_name='Orion Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(target_name='Orion Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(target_name='Orion Pane 4',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 19, 29, 43),
                         dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 12)

    self.context = ray.get(
        _compute_graph_fragment.remote(self.context, filenames))

    stacks = ray.get(self.context.stacks)


if __name__ == "__main__":
  unittest.main()
