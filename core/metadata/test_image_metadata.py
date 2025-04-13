from ray_infra import ray_local

import ray
import glob
import unittest
import random
import os

from context import context
from dependency_graph import run_graph
from testing import make_test_context, make_synthetic_frames
from metadata.image_metadata import scatter_compute_image_metadata, load_metadata


def compute_graph_fragment():
  return {(scatter_compute_image_metadata, 'input'): {}}


def get_all_filenames():
  metadata = ray.get(load_metadata.remote())
  return [i.filename for i in metadata.values()]


def get_metadata():
  return list(ray.get(load_metadata.remote()).values())


class TestImageMetadata(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_existing_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=10,
                          exposure=30,
                          dest_dir=input_dir,
                          insert_metadata=False)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_on_disk = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    self.assertEqual(len(filenames_on_disk), 10)
    filenames_in_cache = get_all_filenames()
    self.assertEqual(sorted(filenames_on_disk), sorted(filenames_in_cache))
    self.assertTrue(all(i.header['EXPOSURE'] == 30 for i in get_metadata()))

  def test_moved_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=20, dest_dir=input_dir, insert_metadata=False)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_on_disk = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    filenames_in_cache = get_all_filenames()
    self.assertEqual(sorted(filenames_on_disk), sorted(filenames_in_cache))
    self.assertTrue(all(os.path.exists(i.filename) for i in get_metadata()))

    new_dir = os.path.join(input_dir, 'new_dir')
    os.mkdir(new_dir)
    random.shuffle(filenames_on_disk)
    for filename in filenames_on_disk[:5]:
      os.rename(filename, os.path.join(new_dir, os.path.basename(filename)))

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_on_disk = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    filenames_in_cache = get_all_filenames()
    self.assertEqual(len(filenames_in_cache), 20)
    self.assertTrue(all(os.path.exists(f) for f in filenames_in_cache))

    self.assertEqual(sum(1 for f in filenames_in_cache if 'new_dir' in f), 5)

  def test_added_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=10,
                          prefix='foo',
                          dest_dir=input_dir,
                          insert_metadata=False)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    self.assertEqual(len(filenames), 10)
    self.assertEqual(len(get_all_filenames()), 10)
    filenames_in_cache = get_all_filenames()
    self.assertEqual(len(filenames_in_cache), 10)
    self.assertEqual(sorted(filenames), sorted(filenames_in_cache))

    make_synthetic_frames(number=10,
                          prefix='bar',
                          dest_dir=input_dir,
                          insert_metadata=False)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    self.assertEqual(len(filenames), 20)
    filenames_in_cache = get_all_filenames()
    self.assertEqual(len(filenames_in_cache), 20)
    self.assertEqual(sorted(filenames), sorted(filenames_in_cache))

  def test_deleted_files(self):
    input_dir = context.input_dirs[0]
    make_synthetic_frames(number=20, dest_dir=input_dir, insert_metadata=False)
    filenames = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    self.assertEqual(len(filenames), 20)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_in_cache = get_all_filenames()
    self.assertEqual(len(filenames_in_cache), 20)

    random.shuffle(filenames)
    for filename in filenames[:5]:
      os.remove(filename)

    filenames = glob.glob(f'{input_dir}/**/*.fit?', recursive=True)
    self.assertEqual(len(filenames), 15)

    graph = compute_graph_fragment()
    run_graph(graph)

    filenames_in_cache = get_all_filenames()
    self.assertEqual(len(filenames_in_cache), 15)


if __name__ == '__main__':
  unittest.main()
