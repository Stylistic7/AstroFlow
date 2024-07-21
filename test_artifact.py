import ray_local

import os
import unittest
import numpy as np

from context import context
from testing.test_context import make_test_context
from testing.test_frames import copy_testdata_frames
from artifact import FileArtifact, CacheArtifact, ArtifactNotFoundException, WrongContentHashException, read_artifact_header, read_artifact_image, write_artifact


class TestMosaic(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_artifact_disk_to_cache_round_trip(self):
    copy_testdata_frames(target_name='NGC7635', dest_dir=context.input_dirs[0])

    filenames = os.listdir(context.input_dirs[0])

    for i, filename in enumerate(filenames):
      full_path = os.path.join(context.input_dirs[0], filename)
      header = read_artifact_header(FileArtifact(full_path))
      image = read_artifact_image(FileArtifact(full_path))

      key = ('image', filename)
      content_hash = f'my_silly_hash_{i}'
      write_artifact(CacheArtifact(key, content_hash), image)

      cached_image = read_artifact_image(CacheArtifact(key, content_hash))
      self.assertTrue(np.array_equal(image, cached_image))

      wrong_content_hash = 'boo'
      with self.assertRaises(WrongContentHashException):
        cached_image = read_artifact_image(
            CacheArtifact(key, wrong_content_hash))

  def test_missing_files(self):
    for i in range(5):
      full_path = f'/my_image_{i}.fit'
      with self.assertRaises(FileNotFoundError):
        header = read_artifact_header(FileArtifact(full_path))

  def test_missing_cache_entries(self):
    for i in range(5):
      key = ('my_image', i)
      with self.assertRaises(ArtifactNotFoundException):
        header = read_artifact_header(CacheArtifact(key))


if __name__ == "__main__":
  unittest.main()
