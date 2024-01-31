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

from make_test_frames import make_test_frames
import ray_diskcache
from astro_flow_direct import Context
from compute_graph import build_compute_graph


class TestMosaic(unittest.TestCase):

  def test_failing_test(self):
    self.assertTrue(False)

  def test_orion_mosaic(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      for pane in range(1, 5):
        for i in range(3):
          make_test_frames(instrument='ASI2600MM Pro',
                           project_dir=tmp_dir,
                           testdata_image=f'orion_pane_{pane}_{i}')

      # make_test_frames(number=5,
      #                  image_type='Dark',
      #                  fill_value=100,
      #                  instrument='ASI2600MM Pro',
      #                  width_pixels=6248,
      #                  height_pixels=4176,
      #                  project_dir=tmp_dir)
      # make_test_frames(number=7,
      #                  image_type='Bias',
      #                  fill_value=50,
      #                  instrument='ASI2600MM Pro',
      #                  width_pixels=6248,
      #                  height_pixels=4176,
      #                  project_dir=tmp_dir)
      # make_test_frames(number=11,
      #                  image_type='Flat',
      #                  fill_value=4096,
      #                  exposure=1,
      #                  instrument='ASI2600MM Pro',
      #                  width_pixels=6248,
      #                  height_pixels=4176,
      #                  project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob("*.fit*")
      ])
      self.assertEqual(len(filenames), 12)

      image_hashes = ray_diskcache.get('image_hashes', {})
      new_filenames = list(set(filenames) - set(image_hashes.values()))

      context = Context({'image_hashes': image_hashes})
      result = ray.workflow.run(build_compute_graph.bind(
          context, new_filenames),
                                workflow_id='AstroFlow')

      stacks = result['stacks']
      self.assertEqual(len(stacks), 2)
      for stack in stacks.values():
        print(f'### stack {stack}')


if __name__ == "__main__":
  unittest.main()
