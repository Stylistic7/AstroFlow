# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import tempfile
import ray
ray.init(
  address='local',
  namespace='AstroFlow',
  resources={"diskcache": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
  storage=tempfile.TemporaryDirectory().name
)

import ray_diskcache
global_var_actor = ray_diskcache.GlobalVarActor.options(name='ray_diskcache',
                                          namespace='AstroFlow',
                                          get_if_exists=True).remote()
# yapf: enable

import unittest
import tempfile
import pathlib

from cache import cache
from make_test_frames import make_test_frames
import new_dependency_graph as dg
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

      image_hashes = ray_diskcache.get('image_hashes', {})
      new_filenames = list(set(filenames) - set(image_hashes.values()))

      result = ray.workflow.run(build_compute_graph.bind(new_filenames),
                                workflow_id='AstroFlow')


if __name__ == "__main__":
  unittest.main()
