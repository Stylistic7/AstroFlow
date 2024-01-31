import ray_local

import unittest
import tempfile
import ray

from cache import cache
from make_test_frames import make_synthetic_frames, copy_testdata_frames
import ray_diskcache

from compute_graph import build_compute_graph
from context import Context
import file_io
import pix


class TestAstroFlow(unittest.TestCase):

  def setUp(self):
    pix._clear_cache()
    ray_diskcache.clear()

    # Start with an empty context.
    self.context = Context()
    self.context.image_hashes = ray.put({})
    self.context.fits_headers = ray.put({})
    self.context.plate_solutions = ray.put({})

    self.input_dir = tempfile.TemporaryDirectory()
    self.context.input_dirs.append(self.input_dir.name)

    self.output_dir = tempfile.TemporaryDirectory()
    self.context.output_dir = self.output_dir.name

  def test_compute_graph(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      copy_testdata_frames(target_name='M42', dest_dir=tmp_dir)
      copy_testdata_frames(target_name='NGC7635', dest_dir=tmp_dir)
      make_synthetic_frames(number=7, image_type='Dark', dest_dir=tmp_dir)
      make_synthetic_frames(number=11, image_type='Bias', dest_dir=tmp_dir)
      make_synthetic_frames(number=13, image_type='Flat', dest_dir=tmp_dir)

      filenames = file_io.find_files(tmp_dir)
      self.assertEqual(len(filenames), 3 + 3 + 7 + 11 + 13)

      image_hashes = ray_diskcache.get('image_hashes', {})
      new_filenames = list(set(filenames) - set(image_hashes.values()))

      result = ray.workflow.run(build_compute_graph.bind(
          self.context, new_filenames),
                                workflow_id='AstroFlow')

      # TODO: Actually check these results (two stacks, one with calibration
      # frames, another without.)


if __name__ == "__main__":
  unittest.main()
