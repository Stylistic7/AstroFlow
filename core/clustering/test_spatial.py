import ray
import os
from datetime import datetime, timedelta

from clustering import cluster_by_image_center
from context import context
from testing.test_frames import make_synthetic_frames
from dependency_graph import Graph, chain, run_graph
from metadata.image_metadata import scatter_compute_image_metadata, load_metadata
from metadata.classes import MetadataDict

def compute_hash_for_filename() -> dict[str, str]:
  metadata = ray.get(load_metadata.remote()).get()
  assert isinstance(metadata, MetadataDict)
  return {
      os.path.basename(i.filename): i.content_hash for i in metadata.values()
  }


def compute_graph_fragment() -> Graph:
  return chain((scatter_compute_image_metadata, 'input'),
               cluster_by_image_center)


def test_one_perfect_group(test_context) -> None:
  input_dir = context.input_dirs[0]
  make_synthetic_frames(ra=100,
                        dec=50,
                        prefix='foo',
                        number=10,
                        dither=0,
                        dest_dir=input_dir)

  results, _ = run_graph(compute_graph_fragment())
  groups = results[cluster_by_image_center]

  metadata = context.get_matching(('compute_image_metadata',))
  image_hashes = [m.content_hash for m in metadata.values()]

  expected_groups = [tuple(sorted(image_hashes))]
  assert groups == expected_groups


def test_two_perfect_groups_one_hour_ra(test_context) -> None:
  input_dir = context.input_dirs[0]

  start_datetime = datetime(2023, 12, 6, 16, 29, 43)
  make_synthetic_frames(start_datetime=start_datetime,
                        ra=300,
                        dec=40,
                        prefix='foo',
                        number=3,
                        dither=0,
                        dest_dir=input_dir)
  make_synthetic_frames(start_datetime=start_datetime + timedelta(hours=1),
                        ra=315,
                        dec=40,
                        prefix='bar',
                        number=3,
                        dither=0,
                        dest_dir=input_dir)

  results, _ = run_graph(compute_graph_fragment())
  groups = results[cluster_by_image_center]

  hash_for_filename = compute_hash_for_filename()

  expected_groups = [
      tuple(
          sorted([
              hash_for_filename['foo_00000.fits'],
              hash_for_filename['foo_00001.fits'],
              hash_for_filename['foo_00002.fits']
          ])),
      tuple(
          sorted([
              hash_for_filename['bar_00000.fits'],
              hash_for_filename['bar_00001.fits'],
              hash_for_filename['bar_00002.fits']
          ]))
  ]
  assert sorted(groups) == sorted(expected_groups)


def test_two_perfect_groups_one_degree_dec(test_context) -> None:
  input_dir = context.input_dirs[0]

  start_datetime = datetime(2023, 12, 6, 16, 29, 43)
  make_synthetic_frames(start_datetime=start_datetime,
                        ra=300,
                        dec=39,
                        prefix='foo',
                        number=3,
                        dither=0,
                        dest_dir=input_dir)
  make_synthetic_frames(start_datetime=start_datetime + timedelta(hours=1),
                        ra=300,
                        dec=40,
                        prefix='bar',
                        number=3,
                        dither=0,
                        dest_dir=input_dir)

  results, _ = run_graph(compute_graph_fragment())
  groups = results[cluster_by_image_center]

  hash_for_filename = compute_hash_for_filename()

  expected_groups = [
      tuple(
          sorted([
              hash_for_filename['foo_00000.fits'],
              hash_for_filename['foo_00001.fits'],
              hash_for_filename['foo_00002.fits']
          ])),
      tuple(
          sorted([
              hash_for_filename['bar_00000.fits'],
              hash_for_filename['bar_00001.fits'],
              hash_for_filename['bar_00002.fits']
          ])),
  ]
  assert sorted(groups) == sorted(expected_groups)


def test_two_perfect_groups_half_degree_dec(test_context) -> None:
  input_dir = context.input_dirs[0]

  start_datetime = datetime(2023, 12, 6, 16, 29, 43)
  make_synthetic_frames(start_datetime=start_datetime,
                        ra=300,
                        dec=39.5,
                        prefix='foo',
                        number=3,
                        dither=0,
                        rotation_angle_deg=0,
                        dest_dir=input_dir)
  make_synthetic_frames(start_datetime=start_datetime + timedelta(hours=1),
                        ra=300,
                        dec=40,
                        prefix='bar',
                        number=3,
                        dither=0,
                        rotation_angle_deg=0,
                        dest_dir=input_dir)

  results, _ = run_graph(compute_graph_fragment())
  groups = results[cluster_by_image_center]

  hash_for_filename = compute_hash_for_filename()

  expected_groups = [
      tuple(
          sorted([
              hash_for_filename['bar_00000.fits'],
              hash_for_filename['bar_00001.fits'],
              hash_for_filename['bar_00002.fits']
          ])),
      tuple(
          sorted([
              hash_for_filename['foo_00000.fits'],
              hash_for_filename['foo_00001.fits'],
              hash_for_filename['foo_00002.fits']
          ])),
  ]
  assert sorted(groups) == sorted(expected_groups)


def test_two_noisy_groups(test_context) -> None:
  input_dir = context.input_dirs[0]

  start_datetime = datetime(2023, 12, 6, 16, 29, 43)
  make_synthetic_frames(start_datetime=start_datetime,
                        ra=307,
                        dec=40,
                        prefix='foo',
                        number=3,
                        dither=0.1,
                        dest_dir=input_dir)
  make_synthetic_frames(start_datetime=start_datetime + timedelta(hours=1),
                        ra=315,
                        dec=45,
                        prefix='bar',
                        number=3,
                        dither=0.1,
                        dest_dir=input_dir)

  results, _ = run_graph(compute_graph_fragment())
  groups = results[cluster_by_image_center]

  hash_for_filename = compute_hash_for_filename()

  expected_groups = [
      tuple(
          sorted([
              hash_for_filename['bar_00000.fits'],
              hash_for_filename['bar_00001.fits'],
              hash_for_filename['bar_00002.fits']
          ])),
      tuple(
          sorted([
              hash_for_filename['foo_00000.fits'],
              hash_for_filename['foo_00001.fits'],
              hash_for_filename['foo_00002.fits']
          ])),
  ]
  assert sorted(groups) == sorted(expected_groups)


def test_sadr_pano(test_context) -> None:
  print('Entering test_sadr_pano')

  input_dir = context.input_dirs[0]

  centers = [
      (301.625, 41.37361111111111),
      (301.81666666666666, 38.29111111111111),
      (304.34583333333336, 41.44222222222222),
      (304.4125, 38.35472222222222),
      (307.06666666666666, 41.44638888888888),
      (307.01666666666665, 38.36083333333333),
  ]
  ids = ['Pane 0', 'Pane 1', 'Pane 2', 'Pane 3', 'Pane 4', 'Pane 5']

  make_synthetic_frames(centers=centers, ids=ids, dither=0, dest_dir=input_dir)

  context.min_component_members = 1

  results, _ = run_graph(compute_graph_fragment())
  groups = results[cluster_by_image_center]

  hash_for_filename = compute_hash_for_filename()
  print(f'### hash_for_filename {hash_for_filename}')

  expected_groups = [(hash_for_filename['Pane 0'],),
                     (hash_for_filename['Pane 1'],),
                     (hash_for_filename['Pane 2'],),
                     (hash_for_filename['Pane 3'],),
                     (hash_for_filename['Pane 4'],),
                     (hash_for_filename['Pane 5'],)]
  assert sorted(groups) == sorted(expected_groups)
