from ray_infra import ray_local

import ray
import unittest

from context import context
from dependency_graph.debugging import Graph, Node, print_graph
from dependency_graph.classes import canonicalize_graph
from testing.test_context import make_test_context
from mosaics.mosaic import Mosaic
from mosaics.placement import place_image
from mosaics.merge_images import merge_images
from mosaics.compute_graph import build_mosaic_filter_subgraph, build_placement_subgraph
from mosaics.types import TileIndices
from previews import QualityLevel
from targets import Target


# TODO: Put this somewhere generally-accessible.
def make_test_target(target,
                     center=[0, 0],
                     image_scale=1.0,
                     rotation_angle=0,
                     footprint=None,
                     projected_footprints=None,
                     content_hash='content_hash',
                     stable_hash='stable_hash',
                     stack_ids=[],
                     oldest_date=None,
                     newest_date=None) -> Target:
  return Target(target=target,
                center=center,
                image_scale=image_scale,
                rotation_angle=rotation_angle,
                footprint=footprint,
                projected_footprints=projected_footprints,
                content_hash=content_hash,
                stable_hash=stable_hash,
                stack_ids=stack_ids,
                oldest_date=oldest_date,
                newest_date=newest_date)


class TestMergeGraph(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self) -> None:
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_two_by_three(self) -> None:
    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None

    mosaic = Mosaic(name='two_by_three',
                    content_hash='content_abc123',
                    stable_hash='mosaic_abc123',
                    target_ids=['a', 'b', 'c', 'd', 'e', 'f'],
                    tile_indices=None,
                    strips=[['a', 'b', 'c'], ['d', 'e', 'f']],
                    oldest_date=None,
                    newest_date=None,
                    center=None,
                    footprint=None,
                    projected_footprints=None)

    context.targets = targets = {
        'a': make_test_target(target='a'),
        'b': make_test_target(target='b'),
        'c': make_test_target(target='c'),
        'd': make_test_target(target='d'),
        'e': make_test_target(target='e'),
        'f': make_test_target(target='f')
    }

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox))
    print_graph(graph)

    expected_graph = Graph({
        Node(merge_images, mode, name='merge_b'): [
            (build_placement_subgraph, mode, mosaic, targets['a'], filter,
             TileIndices((0, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic, targets['b'], filter,
             TileIndices((0, 1)), viewbox)
        ],
        Node(merge_images, mode, name='merge_c'): [
            Node(merge_images, mode, name='merge_b'),
            (build_placement_subgraph, mode, mosaic, targets['c'], filter,
             TileIndices((0, 2)), viewbox)
        ],
        Node(merge_images, mode, name='merge_e'): [
            (build_placement_subgraph, mode, mosaic, targets['d'], filter,
             TileIndices((1, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic, targets['e'], filter,
             TileIndices((1, 1)), viewbox)
        ],
        Node(merge_images, mode, name='merge_f'): [
            Node(merge_images, mode, name='merge_e'),
            (build_placement_subgraph, mode, mosaic, targets['f'], filter,
             TileIndices((1, 2)), viewbox)
        ],
        Node(merge_images, mode, name='merge_strip_1'): [
            Node(merge_images, mode, name='merge_c'),
            Node(merge_images, mode, name='merge_f')
        ]
    })
    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))
