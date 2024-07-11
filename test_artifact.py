import ray_local

import os
import unittest
import numpy as np

from context import context
from make_test_context import make_test_context
from make_test_frames import copy_testdata_frames
from artifact import FileArtifact, CacheArtifact, read_artifact_header, read_artifact_image, write_artifact


class TestMosaic(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_artifact_disk_to_cache_round_trip(self):
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])

    filenames = os.listdir(context.input_dirs[0])

    for filename in filenames:
      full_path = os.path.join(context.input_dirs[0], filename)
      header = read_artifact_header(FileArtifact(full_path))
      image = read_artifact_image(FileArtifact(full_path))

      key = ('image', filename)
      write_artifact(CacheArtifact(key), header, image)

      cached_image = read_artifact_image(CacheArtifact(key))
      self.assertTrue(np.array_equal(image, cached_image))


if __name__ == "__main__":
  unittest.main()
