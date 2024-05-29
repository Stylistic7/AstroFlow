import ray_local

import ray
import glob
import unittest
import tempfile
import random
import os

from context import context
from image_metadata import scatter_compute_image_metadata
from make_test_frames import make_synthetic_frames
from dependency_graph import make_graph_from_list, run_graph


def compute_graph_fragment():
  return make_graph_from_list([(scatter_compute_image_metadata, 'input')])


class TestImageMetadata(unittest.TestCase):

  def setUp(self):
    context.clear()

    # Start with an empty context.
    context.image_hashes = {}
    context.fits_headers = {}
    context.plate_solutions = {}
    context.quarantined_files = []

    self.input_dir = tempfile.TemporaryDirectory()
    context.input_dirs = [self.input_dir.name]

  def test_existing_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=10, exposure=30, dest_dir=input_dir)

    graph = compute_graph_fragment()
    results = run_graph(graph)

    self.assertEqual(results['num_nodes_run'], 2 + 10)
    filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
    image_hashes = context.image_hashes
    fits_headers = context.fits_headers
    plate_solutions = context.plate_solutions
    self.assertEqual(len(filenames), 10)
    self.assertEqual(sorted(filenames), sorted(image_hashes.values()))
    self.assertTrue(all(f['EXPOSURE'] == 30 for f in fits_headers.values()))
    self.assertEqual(sorted(plate_solutions.keys()),
                     sorted(image_hashes.keys()))

  def test_moved_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=20, dest_dir=input_dir)

    graph = compute_graph_fragment()
    results = run_graph(graph)
    self.assertEqual(results['num_nodes_run'], 2 + 20)
    filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
    image_hashes = context.image_hashes
    self.assertEqual(sorted(filenames), sorted(image_hashes.values()))
    self.assertTrue(all(os.path.exists(f) for f in image_hashes.values()))

    new_dir = os.path.join(input_dir, 'new_dir')
    os.mkdir(new_dir)
    random.shuffle(filenames)
    for filename in filenames[:5]:
      os.rename(filename, os.path.join(new_dir, os.path.basename(filename)))

    graph = compute_graph_fragment()
    results = run_graph(graph)
    filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
    image_hashes = context.image_hashes
    self.assertEqual(len(image_hashes), 20)
    self.assertTrue(all(os.path.exists(f) for f in image_hashes.values()))

    self.assertEqual(sum(1 for f in image_hashes.values() if 'new_dir' in f), 5)

  def test_added_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=10, dest_dir=input_dir)

    graph = compute_graph_fragment()
    results = run_graph(graph)
    self.assertEqual(results['num_nodes_run'], 2 + 10)
    filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
    self.assertEqual(len(filenames), 10)
    image_hashes = context.image_hashes
    self.assertEqual(len(image_hashes), 10)
    self.assertEqual(sorted(filenames), sorted(image_hashes.values()))

    make_synthetic_frames(number=10, dest_dir=input_dir)

    graph = compute_graph_fragment()
    results = run_graph(graph)
    self.assertEqual(results['num_nodes_run'], 2 + 10)
    filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
    self.assertEqual(len(filenames), 20)
    image_hashes = context.image_hashes
    self.assertEqual(len(image_hashes), 20)
    self.assertEqual(sorted(filenames), sorted(image_hashes.values()))

  # TODO: What should be done about deleted files? Maybe nothing, until they
  # are actually needed by something, e.g. a stacking operation. They might
  # simply be on removable media, and we should not care until necessary.
  # def test_deleted_files(self):
  #   input_dir = self.context.input_dirs[0]
  #   make_synthetic_frames(number=20, dest_dir=input_dir)

  #   graph = compute_graph_fragment()
  #   self.context, results = run_graph(graph, self.context)
  #   self.assertEqual(results['num_nodes_run'], 2 + 20)
  #   filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
  #   self.assertEqual(len(filenames), 20)
  #   image_hashes = context.image_hashes
  #   self.assertEqual(len(image_hashes), 20)

  #   random.shuffle(filenames)
  #   for filename in filenames[:5]:
  #     os.remove(filename)

  #   graph = compute_graph_fragment()
  #   self.context, results = run_graph(graph, self.context)
  #   self.assertEqual(results['num_nodes_run'], 2)
  #   filenames = glob.glob(f'{input_dir}/**/*.fit', recursive=True)
  #   self.assertEqual(len(filenames), 15)
  #   image_hashes = context.image_hashes
  #   self.assertEqual(len(image_hashes), 20)
  #   self.assertEqual(sum(1 for f in image_hashes.values() if f), 15)


if __name__ == '__main__':
  unittest.main()
