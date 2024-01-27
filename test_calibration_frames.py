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
from compute_graph import build_compute_graph
import image_handling


class TestCalibrationFrames(unittest.TestCase):

  def test_flats_from_different_camera(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      # There are lights flats from two different cameras. This should result
      # in two stacks, with mutually-exclusive sets of flat frames.
      make_test_frames(number=15,
                       instrument='ASI6200MM Pro',
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir,
                       testdata_image='sadr_region')
      make_test_frames(number=23,
                       instrument='ASI2600MM Pro',
                       width_pixels=6248,
                       height_pixels=4176,
                       project_dir=tmp_dir,
                       testdata_image='sadr_region')
      make_test_frames(number=7,
                       image_type='Dark',
                       fill_value=100,
                       instrument='ASI6200MM Pro',
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir)
      make_test_frames(number=11,
                       image_type='Bias',
                       fill_value=50,
                       instrument='ASI6200MM Pro',
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir)
      make_test_frames(number=19,
                       image_type='Flat',
                       fill_value=4096,
                       exposure=1,
                       instrument='ASI6200MM Pro',
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir)
      make_test_frames(number=13,
                       image_type='Flat',
                       fill_value=4096,
                       exposure=1,
                       instrument='ASI2600MM Pro',
                       width_pixels=6248,
                       height_pixels=4176,
                       project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob("*.fit*")
      ])
      self.assertEqual(len(filenames), 88)

      image_hashes = ray_diskcache.get('image_hashes', {})
      new_filenames = list(set(filenames) - set(image_hashes.values()))

      result = ray.workflow.run(build_compute_graph.bind(new_filenames),
                                workflow_id='AstroFlow')

      stacks = result['stacks']
      self.assertEqual(len(stacks), 2)

      stack = ray.get(stacks[0])
      print('### first stack', stack)
      self.assertEqual(len(stack.light_frames), 15)
      self.assertEqual(len(stack.dark_frames), 7)
      self.assertEqual(len(stack.bias_frames), 11)
      self.assertEqual(len(stack.flat_frames), 19)
      self.assertTrue(('stack_thumbnail_image', stack.id,
                       stack.content_hash) in cache)
      self.assertTrue(('stack_thumbnail_video', stack.id,
                       stack.content_hash) in cache)

  def test_dark_scaling(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      # The light frames are 120 seconds, but the dark frames are 60 or 300
      # seconds. AstroFlow should pick the closest darks that are equal to or
      # longer than the lights.
      make_test_frames(
          number=15,
          exposure=120,
          project_dir=tmp_dir,
          testdata_image='sadr_region',
          width_pixels=9576,
          height_pixels=6388,
      )
      make_test_frames(number=7,
                       image_type='Dark',
                       fill_value=100,
                       exposure=300,
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir)
      make_test_frames(number=11,
                       image_type='Bias',
                       fill_value=50,
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir)
      make_test_frames(number=13,
                       image_type='Flat',
                       fill_value=4096,
                       exposure=1,
                       width_pixels=9576,
                       height_pixels=6388,
                       project_dir=tmp_dir)

      filenames = sorted([
          str(path.absolute()) for path in pathlib.Path(tmp_dir).glob("*.fit*")
      ])
      self.assertEqual(len(filenames), 46)

      image_hashes = ray_diskcache.get('image_hashes', {})
      new_filenames = list(set(filenames) - set(image_hashes.values()))

      result = ray.workflow.run(build_compute_graph.bind(new_filenames),
                                workflow_id='AstroFlow')

      stacks = result['stacks']
      self.assertEqual(len(stacks), 1)
      stack = ray.get(stacks[0])

      self.assertEqual(len(stack.light_frames), 15)
      self.assertEqual(len(stack.dark_frames), 7)
      self.assertEqual(len(stack.bias_frames), 11)
      self.assertEqual(len(stack.flat_frames), 13)

      self.assertTrue(('stack_thumbnail_image', stack.id,
                       stack.content_hash) in cache)
      self.assertTrue(('stack_thumbnail_video', stack.id,
                       stack.content_hash) in cache)


if __name__ == "__main__":
  unittest.main()
