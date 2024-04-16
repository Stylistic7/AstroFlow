import ray_local

import unittest
import tempfile
import datetime
import ray
import tarfile
import diskcache
import os

import ray_diskcache
import file_io
from context import Context
from make_test_frames import copy_testdata_frames
import compute_graph
from new_dependency_graph import run_graph
from cache import cache, gzip_cache

TESTDATA_CACHE_GZIP_FILENAME = 'testdata/test_mosaic.gz'


class TestMosaic(unittest.TestCase):

  def setUp(self):
    ray_diskcache.clear()

    # Copy everything from the testdata cache into the "official" cache.
    self.testdata_cache_directory = tempfile.TemporaryDirectory()
    file = tarfile.open(TESTDATA_CACHE_GZIP_FILENAME)
    file.extractall(self.testdata_cache_directory.name)
    file.close()
    self.testdata_cache = diskcache.Cache(
        os.path.join(self.testdata_cache_directory.name, 'cache'))
    for key in self.testdata_cache.iterkeys():
      cache[key] = self.testdata_cache[key]

    print('image_hashes', cache['image_hashes'])

    # Start with an empty context.
    self.context = Context()
    self.context.image_hashes = ray.put({})
    self.context.fits_headers = ray.put({})
    self.context.plate_solutions = ray.put({})

    self.input_dir = tempfile.TemporaryDirectory()
    self.context.input_dirs.append(self.input_dir.name)

    self.output_dir = tempfile.TemporaryDirectory()
    self.context.output_dir = self.output_dir.name

    self.context.cosine_similarity_threshold = 0.99999
    self.context.min_component_members = 3
    self.context.max_run_gap = datetime.timedelta(minutes=15)
    self.context.min_frames_in_run = 3
    self.context.min_frames_in_stack = 3

    self.context.stacks = ray.put({})
    self.context.targets = ray.put({})
    self.context.mosaics = ray.put({})

    self.context.cache_keys = ray.put([])

    self.context.shopping_cart = ray.put({})
    self.context.run_everything = True

  # TODO: This is a comprehensive test that takes more than two hours to
  # complete, which is too slow to iterate on.
  def test_orion_mosaic(self):
    cache.clear()

    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(target_name='Orion Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(target_name='Orion Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=self.context.input_dirs[0])
    copy_testdata_frames(target_name='Orion Pane 4',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 19, 29, 43),
                         dest_dir=self.context.input_dirs[0])

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 12)

    graph = compute_graph.build_compute_graph(filenames)
    self.context, results = run_graph(graph, self.context)

    image_hashes = ray.get(self.context.image_hashes)
    print('### image_hashes', image_hashes)

    fits_headers = ray.get(self.context.fits_headers)
    print('### fits_headers', fits_headers)

    plate_solutions = ray.get(self.context.plate_solutions)
    print('### plate_solutions', plate_solutions)

    stacks = ray.get(self.context.stacks)
    print('### stacks', stacks)

    for key in ray_diskcache.get_keys():
      print(f'### cache key {key}')

    gzip_cache(TESTDATA_CACHE_GZIP_FILENAME)

  def test_all_tasks_already_done(self):
    graph = compute_graph.build_compute_graph(new_filenames=set())
    self.context, results = run_graph(graph, self.context)


if __name__ == "__main__":
  unittest.main()
