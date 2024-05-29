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

from context import context
from make_testdata_cache import TESTDATA_CACHE_GZIP_FILENAME
import file_io
from make_test_frames import copy_testdata_frames
import compute_graph
from dependency_graph import run_graph
from make_test_context import make_test_context


class TestMosaic(unittest.TestCase):

  def setUp(self):
    # Copy everything from the testdata cache into the Ray cache.
    self.testdata_cache_directory = tempfile.TemporaryDirectory()
    file = tarfile.open(TESTDATA_CACHE_GZIP_FILENAME)
    file.extractall(self.testdata_cache_directory.name)
    file.close()
    self.testdata_cache = diskcache.Cache(
        os.path.join(self.testdata_cache_directory.name, 'cache'))
    for key in self.testdata_cache.iterkeys():
      context[key] = self.testdata_cache[key]

    # Start with an empty context.
    self.input_dir_obj, self.output_dir_obj = make_test_context()

    copy_testdata_frames(target_name='Orion Pane 1',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 16, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 2',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 17, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 3',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 18, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)
    copy_testdata_frames(target_name='Orion Pane 4',
                         start_datetime=datetime.datetime(
                             2024, 2, 5, 19, 29, 43),
                         dest_dir=context.input_dirs[0],
                         add_noise=False)

  def test_all_tasks_already_done(self):
    graph = compute_graph.build_compute_graph()
    run_graph(graph)

  def test_everything_after_stacking(self):
    for key in context.get_cache_keys():
      if 'placed_image' in key or 'mosaic_thumbnail_image' in key:
        context.delete(key)

    filenames = file_io.find_image_files(context.input_dirs[0])
    self.assertEqual(len(filenames), 12)

    graph = compute_graph.build_compute_graph()
    run_graph(graph)

    # shutil.copytree(self.output_dir.name, 'testdata/output', dirs_exist_ok=True)

    # TODO: What are good criteria for completion? The addition of new filenames
    # and cache keys, no doubt.


if __name__ == "__main__":
  unittest.main()
