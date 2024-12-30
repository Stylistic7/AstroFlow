from ray_infra import ray_local

import ray
import unittest

from context import context, get_cache
from dependency_graph.debugging import Graph, Node, print_graph
from dependency_graph.classes import canonicalize_graph
from dependency_graph import run_graph
from testing.test_context import make_test_context
from mosaics.mosaic import Mosaic
from mosaics.placement import place_image
from mosaics.merge_images import merge_images
from mosaics.compute_graph import build_mosaic_filter_subgraph, build_placement_subgraph
from mosaics.types import TileIndices
from previews import QualityLevel
from targets import Target
from testing.testdata_cache import copy_testdata_into_cache
from mosaics.viewbox import compute_mosaic_viewbox, Viewbox
from mosaics import identification
from compute_graph.compute_graph import load_metadata
from mosaics.orion_mosaic import add_orion_mosaic_images
import compute_graph


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

  def test_three_by_one(self) -> None:
    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None

    mosaic = Mosaic(name='three_by_one',
                    content_hash='content_abc123',
                    stable_hash='mosaic_abc123',
                    target_ids=['a', 'b', 'c'],
                    tile_indices=None,
                    strips=[['a'], ['b'], ['c']],
                    oldest_date=None,
                    newest_date=None,
                    center=None,
                    footprint=None,
                    projected_footprints=None)

    context.targets = targets = {
        'a': make_test_target(target='a'),
        'b': make_test_target(target='b'),
        'c': make_test_target(target='c'),
    }

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox))
    print_graph(graph)

    expected_graph = Graph({
        Node(merge_images, mode, name='merge_strip_1'): [
            (build_placement_subgraph, mode, mosaic, targets['a'], filter,
             TileIndices((0, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic, targets['b'], filter,
             TileIndices((1, 0)), viewbox)
        ],
        Node(merge_images, mode, name='merge_strip_2'): [
            Node(merge_images, mode, name='merge_strip_1'),
            (build_placement_subgraph, mode, mosaic, targets['c'], filter,
             TileIndices((2, 0)), viewbox)
        ],
    })
    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

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

  def test_three_panes(self) -> None:
    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None

    mosaic = Mosaic(name='three_panes',
                    content_hash='content_abc123',
                    stable_hash='mosaic_abc123',
                    target_ids=['a', 'b', 'c'],
                    tile_indices=None,
                    strips=[['a'], ['b', 'c']],
                    oldest_date=None,
                    newest_date=None,
                    center=None,
                    footprint=None,
                    projected_footprints=None)

    context.targets = targets = {
        'a': make_test_target(target='a'),
        'b': make_test_target(target='b'),
        'c': make_test_target(target='c'),
    }

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox))

    expected_graph = Graph({
        Node(merge_images, mode, name='merge_c'): [
            (build_placement_subgraph, mode, mosaic, targets['b'], filter,
             TileIndices((1, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic, targets['c'], filter,
             TileIndices((1, 1)), viewbox)
        ],
        Node(merge_images, mode, name='merge_strip_1'): [
            (build_placement_subgraph, mode, mosaic, targets['a'], filter,
             TileIndices((0, 0)), viewbox),
            Node(merge_images, mode, name='merge_c')
        ]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_orion_mosaic(self) -> None:
    add_orion_mosaic_images()

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    targets = context.targets
    self.assertEqual(len(targets), 4)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    self.assertEqual(mosaic.strips, [['e1f7d909ac125ebe', '3871323c8756dead'],
                                     ['3ef21a508ef46b36', '9ed89ccd9f10cc09']])

    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox))

    expected_graph = Graph({
        Node(merge_images, mode, name='merge_3871323c8756dead'): [
            (build_placement_subgraph, mode, mosaic,
             targets['e1f7d909ac125ebe'], filter, TileIndices((0, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic,
             targets['3871323c8756dead'], filter, TileIndices((0, 1)), viewbox)
        ],
        Node(merge_images, mode, name='merge_9ed89ccd9f10cc09'): [
            (build_placement_subgraph, mode, mosaic,
             targets['3ef21a508ef46b36'], filter, TileIndices((1, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic,
             targets['9ed89ccd9f10cc09'], filter, TileIndices((1, 1)), viewbox)
        ],
        Node(merge_images, mode, name='merge_strip_1'): [
            Node(merge_images, mode, name='merge_3871323c8756dead'),
            Node(merge_images, mode, name='merge_9ed89ccd9f10cc09')
        ]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_orion_mosaic_missing_panel(self) -> None:
    add_orion_mosaic_images(panes=[1, 2, 3])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = compute_graph.build_compute_graph(regroup=True,
                                              make_previews=True,
                                              run_main_processing_graph=True)
    run_graph(graph)

    targets = context.targets
    self.assertEqual(len(targets), 3)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    self.assertEqual(
        mosaic.strips,
        [['3ef21a508ef46b36'], ['e1f7d909ac125ebe', '3871323c8756dead']])

    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox))

    expected_graph = Graph({
        Node(merge_images, mode, name='merge_3871323c8756dead'): [
            (build_placement_subgraph, mode, mosaic,
             targets['e1f7d909ac125ebe'], filter, TileIndices((1, 0)), viewbox),
            (build_placement_subgraph, mode, mosaic,
             targets['3871323c8756dead'], filter, TileIndices((1, 1)), viewbox)
        ],
        Node(merge_images, mode, name='merge_strip_1'): [
            (build_placement_subgraph, mode, mosaic,
             targets['3ef21a508ef46b36'], filter, TileIndices((0, 0)), viewbox),
            Node(merge_images, mode, name='merge_3871323c8756dead')
        ]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))
