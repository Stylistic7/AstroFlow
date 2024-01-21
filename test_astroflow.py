# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import ray
ray.init(
  address='local',
  namespace='AstroFlow',
  resources={"diskcache": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
)
# yapf: enable

import time
import unittest
import tempfile
import pathlib

from cache import cache
import file_io
import image_metadata
from make_test_frames import make_test_frames
import new_dependency_graph as dg
import spatial_association
import ray_diskcache

from image_metadata import compute_image_metadata_ray, store_image_metadata_ray
from spatial_association import find_components_by_cosine_similarity_ray
from stack_identification import identify_stacks_ray
from compute_graph import build_compute_graph


class TestAstroFlow(unittest.TestCase):

  def setUp(self):
    ray_diskcache.clear()
    cache.clear()

  def tearDown(self):
    ray_diskcache.clear()
    cache.clear()

  def test_read_fits_headers(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=3, project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob("*.fit*")
      ])

      result = ray.get(image_metadata.read_fits_headers_ray.remote(filenames))

      self.assertEqual(type(result), dg.ResultDict)
      self.assertEqual(len(result.result), 3)

      ray_diskcache.clear()
      cache.clear()

  def test_compute_graph(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=3,
                       project_dir=tmp_dir,
                       testdata_image='bubble_nebula')
      make_test_frames(number=5,
                       project_dir=tmp_dir,
                       testdata_image='sadr_region')
      make_test_frames(number=7, image_type='Dark', project_dir=tmp_dir)
      make_test_frames(number=11, image_type='Bias', project_dir=tmp_dir)
      make_test_frames(number=13, image_type='Flat', project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob("*.fit*")
      ])
      self.assertEqual(len(filenames), 39)

      graph = build_compute_graph(filenames)
      results = dg.run_graph(graph)

      self.assertEqual(len(ray_diskcache.get('image_hashes', {})), 39)
      self.assertEqual(len(ray_diskcache.get('fits_headers', {})), 39)
      self.assertEqual(len(ray_diskcache.get('plate_solutions', {})), 39)

      stacks = ray_diskcache.get('stacks', {})
      self.assertEqual(len(stacks), 2)

      for stack in stacks:
        self.assertEqual(len(stack.camera_configs), 1)
        self.assertEqual(len(stack.dark_frames), 7)
        self.assertEqual(len(stack.bias_frames), 11)
        self.assertEqual(len(stack.flat_frames), 13)

        for light_frame in stack.light_frames:
          self.assertTrue(('image_thumbnail', light_frame) in cache)


if __name__ == "__main__":
  unittest.main()
