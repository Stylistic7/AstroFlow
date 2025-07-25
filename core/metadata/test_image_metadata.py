import ray
import glob
import random
import os
from pathlib import Path

from context import context
from dependency_graph import run_graph
from testing import make_test_context, make_synthetic_frames
from metadata.image_metadata import scatter_compute_image_metadata, load_metadata


def compute_graph_fragment():
  return {(scatter_compute_image_metadata, 'input'): {}}


def get_filenames_in_cache():
  metadata = ray.get(load_metadata.remote())
  return [i.filename for i in metadata.values()]


def get_filenames_on_disk(input_dir: Path) -> list[Path]:
  return [Path(f) for f in glob.glob(f'{input_dir}/**/*.fit?', recursive=True)]


def get_metadata():
  return list(ray.get(load_metadata.remote()).values())


def test_existing_files(test_context):
  input_dir = context.input_dirs[0]
  make_synthetic_frames(number=3,
                        exposure=30,
                        dest_dir=input_dir,
                        insert_metadata=False)

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames_on_disk = get_filenames_on_disk(input_dir)
  assert len(filenames_on_disk) == 3
  filenames_in_cache = get_filenames_in_cache()
  assert sorted(filenames_on_disk), sorted(filenames_in_cache)
  assert all(i.header['EXPOSURE'] == 30 for i in get_metadata())


def test_moved_files(test_context):
  input_dir = context.input_dirs[0]
  make_synthetic_frames(number=5, dest_dir=input_dir, insert_metadata=False)

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames_on_disk = get_filenames_on_disk(input_dir)
  filenames_in_cache = get_filenames_in_cache()
  assert sorted(filenames_on_disk) == sorted(filenames_in_cache)
  assert all(os.path.exists(i.filename) for i in get_metadata())

  new_dir = os.path.join(input_dir, 'new_dir')
  os.mkdir(new_dir)
  random.shuffle(filenames_on_disk)
  for filename in filenames_on_disk[:2]:
    os.rename(filename, os.path.join(new_dir, os.path.basename(filename)))

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames_on_disk = get_filenames_on_disk(input_dir)
  filenames_in_cache = get_filenames_in_cache()
  assert len(filenames_in_cache) == 5
  assert all(os.path.exists(f) for f in filenames_in_cache)

  assert sum(1 for f in filenames_in_cache if 'new_dir' in str(f)) == 2


def test_added_files(test_context):
  input_dir = context.input_dirs[0]
  make_synthetic_frames(number=3,
                        prefix='foo',
                        dest_dir=input_dir,
                        insert_metadata=False)

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames = get_filenames_on_disk(input_dir)
  assert len(filenames) == 3
  assert len(get_filenames_in_cache()) == 3
  filenames_in_cache = get_filenames_in_cache()
  assert len(filenames_in_cache) == 3
  assert sorted(filenames) == sorted(filenames_in_cache)

  make_synthetic_frames(number=5,
                        prefix='bar',
                        dest_dir=input_dir,
                        insert_metadata=False)

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames = get_filenames_on_disk(input_dir)
  assert len(filenames) == 8
  filenames_in_cache = get_filenames_in_cache()
  assert len(filenames_in_cache) == 8
  assert sorted(filenames) == sorted(filenames_in_cache)


def test_deleted_files(test_context):
  input_dir = context.input_dirs[0]
  make_synthetic_frames(number=3, dest_dir=input_dir, insert_metadata=False)
  filenames = get_filenames_on_disk(input_dir)
  assert len(filenames) == 3

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames_in_cache = get_filenames_in_cache()
  assert len(filenames_in_cache) == 3

  random.shuffle(filenames)
  for filename in filenames[:2]:
    os.remove(filename)

  filenames = get_filenames_on_disk(input_dir)
  assert len(filenames) == 1

  graph = compute_graph_fragment()
  run_graph(graph)

  filenames_in_cache = get_filenames_in_cache()
  assert len(filenames_in_cache) == 1
