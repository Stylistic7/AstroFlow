import ray_local

import unittest
import tempfile
import ray

from make_test_frames import make_test_frames
import stacking
import file_io
import image_metadata
from context import Context
import ray_diskcache


class TestStacking(unittest.TestCase):

  def setUp(self):
    pix._clear_cache()
    ray_diskcache.clear()

    # Start with an empty context.
    self.context = Context()
    self.context.image_hashes = ray.put({})
    self.context.fits_headers = ray.put({})
    self.context.plate_solutions = ray.put({})

    self.context.self.context.input_dirs.append(tempfile.TemporaryDirectory())
    self.context.self.context.output_dir = tempfile.TemporaryDirectory()

  def test_master_dark(self):
    # Create a set of dark frames.
    make_test_frames(number=5,
                     image_type='Dark',
                     fill_value=100,
                     instrument='ASI6200MM Pro',
                     width_pixels=9576,
                     height_pixels=6388,
                     project_dir=self.context.input_dir)

    # Get their filenames.
    filenames = file_io.find_files(self.context.input_dir)
    self.assertEqual(len(filenames), 5)

    # Compute image metadata for all the dark frames.
    context = ray.workflow.run(
        image_metadata.scatter_compute_image_metadata.bind(context, filenames))
    self.assertEqual(len(ray.get(context.image_hashes)), 5)

    # Stack the dark frames by themselves into a single master.
    dark_hashes = ray.get(context.image_hashes).keys()
    masters = ray.get(
        stacking.stack_files.remote(context, dark_hashes,
                                    self.context.output_dir,
                                    {'camera': 'ASI6200MM Pro'}))
    self.assertEqual(len(masters), 1)

    # Ensure a single master file is in the output dir.
    filenames = file_io.find_files(self.context.output_dir)
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
        image_metadata.scatter_compute_image_metadata.bind(context, filenames))
    self.assertEqual(len(ray.get(context.image_hashes)), 6)


if __name__ == "__main__":
  unittest.main()
