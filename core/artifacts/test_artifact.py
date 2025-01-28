from ray_infra import ray_local

import os
import unittest
import numpy as np

from testing.test_context import make_test_context
from artifacts import FileArtifact, CacheArtifact, ArtifactNotFoundException, read_artifact_header


class TestMosaic(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_missing_files(self) -> None:
    for i in range(5):
      full_path = f'/my_image_{i}.fits'
      with self.assertRaises(FileNotFoundError):
        header = read_artifact_header(FileArtifact(full_path, 'foo'))

  def test_missing_cache_entries(self) -> None:
    for i in range(5):
      key = ('my_image', i)
      with self.assertRaises(ArtifactNotFoundException):
        header = read_artifact_header(CacheArtifact(key, 'foo'))


if __name__ == "__main__":
  unittest.main()
