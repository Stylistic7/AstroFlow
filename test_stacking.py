import unittest
import os
import ray
import graphlib
import tempfile
import functools

from make_test_frames import make_test_frames
import stacking
import new_dependency_graph as dg
from stack_identification import identify_stacks
from spatial_association import find_components_by_cosine_similarity

# This Ray cluster will be able to run tasks that are explicitly local, as well
# as any task with no specific resources.
ray.init(resources={'diskcache': 1})


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
      stack.store_metadata()

      should_run_p = functools.partial(stacking.should_run, image_hashes,
                                       fits_headers, plate_solutions, tmp_dir)

      A = (stacking.should_run, stack)
      B = (stacking.run, stack)

      graph = {B: {A}}
      results = dg.run_graph(graph)


if __name__ == "__main__":
  unittest.main()