import ray_local

import ray
import datetime
import tarfile
import tempfile
import shutil

import file_io
import ray_diskcache
from cache import cache
from make_test_frames import copy_testdata_frames
import compute_graph
from dependency_graph import run_graph
from context import Context
from make_test_context import make_test_context

TESTDATA_CACHE_GZIP_FILENAME = 'testdata/cache.gz'


def gzip_cache(dest_filename):
  tar = tarfile.open(dest_filename, "w:gz")
  tar.add(cache.directory, arcname="cache")
  tar.close()


if __name__ == '__main__':
  self.context, self.input_dir_obj, self.output_dir_obj = make_test_context()
  context.run_everything = True

  # Add four stacks of images that comprise a mosaic of Orion. Each stack
  # contains three images.
  copy_testdata_frames(target_name='Orion Pane 1',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 2',
                       start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 3',
                       start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 4',
                       start_datetime=datetime.datetime(2024, 2, 5, 19, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)

  # Run the complete compute graph, which may take a long time.
  graph = compute_graph.build_compute_graph()
  run_graph(graph, context)

  # Copy the output files, and store the cache to a tar.gz file on disk.
  shutil.copytree(context.output_dir, 'testdata', dirs_exist_ok=True)
  gzip_cache(TESTDATA_CACHE_GZIP_FILENAME)
