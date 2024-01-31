# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import tempfile
import ray
ray.init(
  address='local',
  namespace='AstroFlow',
  resources={"diskcache": 1, "PixInsight": 1},
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

import os
import pathlib
import unittest

from make_test_frames import make_test_frames
import stacking
import file_io
import image_metadata
from context import Context


class TestStacking(unittest.TestCase):

  def test_master_dark(self):
    with tempfile.TemporaryDirectory() as input_dir:
      with tempfile.TemporaryDirectory() as output_dir:
        # Create a set of dark frames.
        make_test_frames(number=5,
                         image_type='Dark',
                         fill_value=100,
                         instrument='ASI6200MM Pro',
                         width_pixels=9576,
                         height_pixels=6388,
                         project_dir=input_dir)

        # Get their filenames.
        filenames = file_io.find_files(input_dir)
        self.assertEqual(len(filenames), 5)

        # Start with an empty context.
        context = Context()
        context.image_hashes = ray.put({})
        context.fits_headers = ray.put({})
        context.plate_solutions = ray.put({})

        # Compute image metadata for all the dark frames.
        context = ray.workflow.run(
            image_metadata.scatter_compute_image_metadata.bind(
                context, filenames))
        self.assertEqual(len(ray.get(context.image_hashes)), 5)

        # Stack the dark frames by themselves into a single master.
        dark_hashes = ray.get(context.image_hashes).keys()
        masters = ray.get(
            stacking.stack_files.remote(context, dark_hashes, output_dir,
                                        {'camera': 'ASI6200MM Pro'}))
        self.assertEqual(len(masters), 1)

        # Ensure a single master file is in the output dir.
        filenames = file_io.find_files(output_dir)
        self.assertEqual(len(filenames), 1)

        # Load the metadata for the master dark, and make sure it carries
        # through the metadata given above, and contains a content hash. The
        # content hash uniquely defines the set of dark frames that went into
        # the master dark.
        filename = filenames[0]
        metadata = file_io.read_header(filename)
        self.assertTrue('camera' in metadata)
        self.assertEqual(metadata['camera'], 'ASI6200MM Pro')
        self.assertTrue(stacking._CONTENT_HASH_KEY in metadata)

        # Compute the metadata of the master. Ensure that the
        context = ray.workflow.run(
            image_metadata.scatter_compute_image_metadata.bind(
                context, filenames))
        self.assertEqual(len(ray.get(context.image_hashes)), 6)


if __name__ == "__main__":
  unittest.main()
