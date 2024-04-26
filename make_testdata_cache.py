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
from new_dependency_graph import run_graph
from context import Context

TESTDATA_CACHE_GZIP_FILENAME = 'testdata/cache.gz'


def gzip_cache(dest_filename):
  tar = tarfile.open(dest_filename, "w:gz")
  tar.add(cache.directory, arcname="cache")
  tar.close()


if __name__ == '__main__':
  cache.clear()
  ray_diskcache.clear()

  # Create a Context object and populate it.
  context = Context()
  context.image_hashes = ray.put({})
  context.fits_headers = ray.put({})
  context.plate_solutions = ray.put({})
  context.cache_keys = ray.put([])
  context.quarantined_files = ray.put([])

  input_dir = tempfile.TemporaryDirectory()
  context.input_dirs.append(input_dir.name)

  output_dir = tempfile.TemporaryDirectory()
  context.output_dir = output_dir.name

  context.cosine_similarity_threshold = 0.99999
  context.min_component_members = 3
  context.max_run_gap = datetime.timedelta(minutes=15)
  context.min_frames_in_run = 3
  context.min_frames_in_stack = 3

  context.run_everything = True

  # Add four stacks of images that comprise a mosaic of Orion. Each stack
  # contains three images.
  copy_testdata_frames(target_name='Orion Pane 1',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=input_dir.name,
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 2',
                       start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
                       dest_dir=input_dir.name,
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 3',
                       start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
                       dest_dir=input_dir.name,
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 4',
                       start_datetime=datetime.datetime(2024, 2, 5, 19, 29, 43),
                       dest_dir=input_dir.name,
                       add_noise=False)

  # Run the complete compute graph, which may take a long time.
  graph = compute_graph.build_compute_graph()
  run_graph(graph, context)

  # Copy the output files, and store the cache to a tar.gz file on disk.
  shutil.copytree(output_dir.name, 'testdata', dirs_exist_ok=True)
  gzip_cache(TESTDATA_CACHE_GZIP_FILENAME)
