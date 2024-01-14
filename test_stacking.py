import unittest
import os
import graphlib
import tempfile
import functools

import ray

ray.init(resources={'diskcache': 1}, ignore_reinit_error=True)

from make_test_frames import make_test_frames
import stacking
import new_dependency_graph as dg
from stack_identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity


class TestStacking(unittest.TestCase):
  def test_simple_stack(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      image_hashes, fits_headers, plate_solutions = make_test_frames(
          number=10,
          prefix='foo',
          project_dir=tmp_dir,
          ra=179,
          dec=0,
          rotation_angle_deg=0,
          dither=0,
          fill_value=50)

      spatial_components = find_components_by_cosine_similarity(
          image_hashes.keys(), plate_solutions, fits_headers)
      stacks = identify_stacks(spatial_components, plate_solutions,
                               fits_headers)
      self.assertEqual(len(stacks), 1)
      stack = stacks[0]
      stack.store_metadata(fits_headers, plate_solutions)

      image_hashes = ray.put(image_hashes)
      fits_headers = ray.put(fits_headers)
      plate_solutions = ray.put(plate_solutions)

      A = (stacking.should_run, (image_hashes, fits_headers, plate_solutions,
                                 tmp_dir, stack))
      graph = {A: {}}
      results = dg.run_graph(graph)

      print('### results', results)


if __name__ == "__main__":
  unittest.main()