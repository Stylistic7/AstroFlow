from ray_infra import ray_local

import os
import unittest
from datetime import datetime, timedelta
import ray

from testing.test_context import make_test_context
from context import context
from testing.test_frames import make_synthetic_frames
from clustering import cluster_by_image_center
from metadata.actor import get_metadata
from dependency_graph import Graph, make_graph_from_list, run_graph
from metadata.image_metadata import scatter_compute_image_metadata
from compute_graph.compute_graph import load_metadata


def compute_hash_for_filename() -> dict[str, str]:
  return get_metadata(transform_fn=lambda i:
                      (os.path.basename(i.filename), i.content_hash),
                      post_fn=lambda images: {
                          f: h for f, h in images
                      })


def compute_graph_fragment() -> Graph:
  return make_graph_from_list([(scatter_compute_image_metadata, 'input'),
                               load_metadata, cluster_by_image_center])


class TestSpatialAssociation(unittest.TestCase):

  def test_one_perfect_group(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()
    input_dir = context.input_dirs[0]
    make_synthetic_frames(ra=100,
                          dec=50,
                          prefix='foo',
                          number=10,
                          dither=0,
                          dest_dir=input_dir)

    results, _ = run_graph(compute_graph_fragment())
    groups = results[cluster_by_image_center]

    image_hashes = get_metadata(transform_fn=lambda i: i.content_hash)

    expected_groups = [tuple(sorted(image_hashes))]
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_one_hour_ra(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()
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
                hash_for_filename['foo_00000.fit'],
                hash_for_filename['foo_00001.fit'],
                hash_for_filename['foo_00002.fit']
            ])),
        tuple(
            sorted([
                hash_for_filename['bar_00000.fit'],
                hash_for_filename['bar_00001.fit'],
                hash_for_filename['bar_00002.fit']
            ]))
    ]
    self.assertEqual(sorted(groups), sorted(expected_groups))

  def test_two_perfect_groups_one_degree_dec(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()
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
                hash_for_filename['foo_00000.fit'],
                hash_for_filename['foo_00001.fit'],
                hash_for_filename['foo_00002.fit']
            ])),
        tuple(
            sorted([
                hash_for_filename['bar_00000.fit'],
                hash_for_filename['bar_00001.fit'],
                hash_for_filename['bar_00002.fit']
            ])),
    ]
    self.assertEqual(sorted(groups), sorted(expected_groups))

  def test_two_perfect_groups_half_degree_dec(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()
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
                hash_for_filename['bar_00000.fit'],
                hash_for_filename['bar_00001.fit'],
                hash_for_filename['bar_00002.fit']
            ])),
        tuple(
            sorted([
                hash_for_filename['foo_00000.fit'],
                hash_for_filename['foo_00001.fit'],
                hash_for_filename['foo_00002.fit']
            ])),
    ]
    self.assertEqual(sorted(groups), sorted(expected_groups))

  def test_two_noisy_groups(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()
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
                hash_for_filename['bar_00000.fit'],
                hash_for_filename['bar_00001.fit'],
                hash_for_filename['bar_00002.fit']
            ])),
        tuple(
            sorted([
                hash_for_filename['foo_00000.fit'],
                hash_for_filename['foo_00001.fit'],
                hash_for_filename['foo_00002.fit']
            ])),
    ]
    self.assertEqual(sorted(groups), sorted(expected_groups))

  def test_sadr_pano(self) -> None:
    print('Entering test_sadr_pano')

    self.input_dir_obj, self.output_dir_obj = make_test_context()
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

    make_synthetic_frames(centers=centers,
                          ids=ids,
                          dither=0,
                          dest_dir=input_dir)

    context.min_component_members = 1

    results, _ = run_graph(compute_graph_fragment())
    groups = results[cluster_by_image_center]

    hash_for_filename = compute_hash_for_filename()

    expected_groups = [(hash_for_filename['Pane 0'],),
                       (hash_for_filename['Pane 1'],),
                       (hash_for_filename['Pane 2'],),
                       (hash_for_filename['Pane 3'],),
                       (hash_for_filename['Pane 4'],),
                       (hash_for_filename['Pane 5'],)]
    self.assertEqual(sorted(groups), sorted(expected_groups))
