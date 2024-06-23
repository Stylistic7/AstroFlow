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
from hashing import hash_list


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

  def test_merge_mosaic(self):
    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = compute_graph.build_compute_graph()
    run_graph(graph)

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    key = ('mosaic_thumbnail_image', 'subframe', mosaic.stable_hash, 'H',
           mosaic.content_hash)
    self.assertTrue(context.contains(key))

    self.assertTrue(
        mosaic.stable_hash in [t.stable_hash for t in context.targets.values()])
    self.assertEqual(
        mosaic.content_hash,
        hash_list([t.content_hash for t in context.targets.values()]))

    # After putting the mosaic in the shopping cart, the second graph
    # execution should stack the subframes into masters, and generate a final
    # master mosaic image.
    context.shopping_cart = [('mosaic', mosaic.stable_hash)]

    # TODO: Put per-mosaic overrides of the post-processing scripts, to turn
    # them off, for speed.

    graph = compute_graph.build_compute_graph()
    run_graph(graph)

    self.assertEqual(len(context.targets), 4)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))


if __name__ == "__main__":
  unittest.main()
