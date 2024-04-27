import ray_local

import unittest
import tempfile
import datetime
import ray
import tarfile
import diskcache
import os
from pathlib import Path
import shutil

import ray_diskcache
from make_testdata_cache import TESTDATA_CACHE_GZIP_FILENAME
import file_io
from context import Context
from make_test_frames import copy_testdata_frames
import compute_graph
from new_dependency_graph import run_graph


class TestMosaic(unittest.TestCase):

  def setUp(self):
    # Copy everything from the testdata cache into the Ray cache.
    self.testdata_cache_directory = tempfile.TemporaryDirectory()
    file = tarfile.open(TESTDATA_CACHE_GZIP_FILENAME)
    file.extractall(self.testdata_cache_directory.name)
    file.close()
    self.testdata_cache = diskcache.Cache(
        os.path.join(self.testdata_cache_directory.name, 'cache'))
    ray_diskcache.clear()
    for key in self.testdata_cache.iterkeys():
      ray_diskcache.set(key, self.testdata_cache[key])
      print(f'### added ray_diskcache key {key}')

    # Start with an empty context.
    self.context = Context()
    self.context.image_hashes = ray.put({})
    self.context.fits_headers = ray.put({})
    self.context.plate_solutions = ray.put({})
    self.context.quarantined_files = ray.put([])

    self.input_dir = tempfile.TemporaryDirectory()
    self.context.input_dirs.append(self.input_dir.name)

    self.output_dir = tempfile.TemporaryDirectory()
    self.context.output_dir = self.output_dir.name
    shutil.copytree('testdata/output', self.output_dir.name, dirs_exist_ok=True)

    self.context.cosine_similarity_threshold = 0.99999
    self.context.min_component_members = 3
    self.context.max_run_gap = datetime.timedelta(minutes=15)
    self.context.min_frames_in_run = 3
    self.context.min_frames_in_stack = 3

    # TODO: The ray.put(ray_diskcache.get()) idiom is weird.
    self.context.cache_keys = ray.put(ray_diskcache.get_keys())

    self.context.shopping_cart = ray.put({})
    self.context.run_everything = True

    image_hashes = ray_diskcache.get('image_hashes', {})
    self.context.image_hashes = ray.put(image_hashes)

    fits_headers = ray_diskcache.get('fits_headers', {})
    self.context.fits_headers = ray.put(fits_headers)

    plate_solutions = ray_diskcache.get('plate_solutions', {})
    self.context.plate_solutions = ray.put(plate_solutions)

    for h, p in plate_solutions.items():
      print(f'### plate solution exists for '
            f'filename {image_hashes[h]}')

    self.context.stacks = ray.put(ray_diskcache.get('stacks', {}))
    self.context.targets = ray.put(ray_diskcache.get('targets', {}))
    self.context.mosaics = ray.put(ray_diskcache.get('mosaics', {}))

    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=self.context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=self.context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=self.context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 4',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 19, 29, 43),
                         dest_dir=self.context.input_dirs[0],
                         add_noise=False)

  def test_all_tasks_already_done(self):
    graph = compute_graph.build_compute_graph()
    self.context, results = run_graph(graph, self.context)

  def test_everything_after_stacking(self):
    for key in ray_diskcache.get_keys():
      if 'placed_image' in key or 'mosaic_thumbnail_image' in key:
        ray_diskcache.delete(key)

    filenames = file_io.find_image_files(self.context.input_dirs[0])
    self.assertEqual(len(filenames), 12)

    graph = compute_graph.build_compute_graph()
    self.context, results = run_graph(graph, self.context)

    shutil.copytree(self.output_dir.name, 'testdata/output', dirs_exist_ok=True)

    # TODO: What are good criteria for completion? The addition of new filenames
    # and cache keys, no doubt.


if __name__ == "__main__":
  unittest.main()
