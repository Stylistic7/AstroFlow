import unittest
import tempfile
import os
import pathlib

import ray

os.environ['ASTROFLOW_MOCK_CACHE'] = "true"

ray.init(resources={'diskcache': 1},
         ignore_reinit_error=True,
         runtime_env={"env_vars": {
             "ASTROFLOW_MOCK_CACHE": "true"
         }})

from cache import cache
import file_io
import image_metadata
from make_test_frames import make_test_frames
import new_dependency_graph as dg
import spatial_association
import ray_diskcache


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
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob('*.fit*')
      ])

      result = ray.get(image_metadata.read_fits_headers_ray.remote(filenames))

      self.assertEqual(type(result), dg.ResultDict)
      self.assertEqual(len(result.result), 3)

      ray_diskcache.clear()
      cache.clear()

  def test_up_to_plate_solving(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=20, project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob('*.fit*')
      ])

      A = (image_metadata.read_fits_headers_ray, tuple(filenames))
      B = (spatial_association.find_groups_by_fits_header_ray,
           dg.FromAncestors)

      graph = {B: {A}}

      results = dg.run_graph(graph)

      result = results[('read_fits_headers_ray', tuple(filenames))]
      self.assertEqual(type(result), dg.ResultDict)
      self.assertTrue(result.ran)
      self.assertEqual(len(result.result), 20)

      # TODO: This is messy. What's the best way to get results after the graph
      # has run?
      result = next(results[k] for k in results
                    if k[0] == 'find_groups_by_fits_header_ray')
      groups = result.result
      self.assertEqual(len(groups), 1)
      self.assertEqual(len(groups[0]), 20)

  def test_including_plate_solving(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      make_test_frames(number=20, project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob('*.fit*')
      ])

      A = (image_metadata.read_fits_headers_ray, tuple(filenames))
      B = (spatial_association.find_groups_by_fits_header_ray,
           dg.FromAncestors)

      graph = {B: {A}}

      for f in filenames:
        graph[(image_metadata.compute_image_metadata_ray, f)] = {B}

      results = dg.run_graph(graph)

      for k in results.keys():
        print(f'### results key {k}')


if __name__ == '__main__':
  unittest.main()
