from ray_infra import ray_local

import ray
import unittest
import datetime
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from compute_graph import build_compute_graph
from context import context
from previews import QualityLevel
from testing import make_test_context

from customization.types import CustomizedStack
from dependency_graph import Graph, Node, print_graph, run_graph, canonicalize_graph
from mosaics.types import Mosaic
from mosaics.merge_images import merge_images
from mosaics.compute_graph import build_mosaic_filter_subgraph, build_placement_subgraph
from mosaics.types import TileIndices
from mosaics.orion_mosaic import add_orion_mosaic_images
from targets.types import Target
from testing.test_stacks import make_composite_stack

_OLDEST_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_NEWEST_DATE = datetime.datetime(2024, 2, 5, 17, 29, 43)
_FOOTPRINT = SingleSphericalPolygon.from_radec([98.01, 102.01, 102.01, 98.01],
                                               [-1, -1, 1, 1],
                                               center=[100.01, 0])


# TODO: Put this somewhere generally-accessible.
def make_test_target(name,
                     center=[0.0, 0.0],
                     image_scale=1.0,
                     rotation_angle=0.0,
                     footprint=None,
                     projected_footprints=None,
                     content_hash='content_hash',
                     stable_hash='stable_hash',
                     stack_ids=[],
                     oldest_date=_OLDEST_DATE,
                     newest_date=_NEWEST_DATE) -> Target:
  return Target(name=name,
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
    masks = None

    mosaic = Mosaic(name='three_by_one',
                    content_hash='content_abc123',
                    stable_hash='mosaic_abc123',
                    target_ids=['a', 'b', 'c'],
                    strips=[['a'], ['b'], ['c']],
                    oldest_date=_OLDEST_DATE,
                    newest_date=_NEWEST_DATE,
                    center=[0.0, 0.0],
                    image_scale=1.0,
                    rotation_angle=0.0,
                    footprint=_FOOTPRINT,
                    projected_footprints={})

    context.targets = targets = {
        'a': make_test_target('a', stack_ids=['a']),
        'b': make_test_target('b', stack_ids=['b']),
        'c': make_test_target('c', stack_ids=['c']),
    }

    context.stacks = stacks = {
        'a': make_composite_stack(stable_hash='a'),
        'b': make_composite_stack(stable_hash='b'),
        'c': make_composite_stack(stable_hash='c'),
    }

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    expected_graph = Graph({
        Node(merge_images, mode, mosaic, name='merge_mosaic_abc123_strip_1'): [
            Node(build_placement_subgraph, mode, mosaic, stacks['a'], filter,
                 TileIndices((0, 0)), viewbox, None),
            Node(build_placement_subgraph, mode, mosaic, stacks['b'], filter,
                 TileIndices((1, 0)), viewbox, None)
        ],
        Node(merge_images, mode, mosaic, name='merge_mosaic_abc123_strip_2'): [
            Node(merge_images, mode, mosaic,
                 name='merge_mosaic_abc123_strip_1'),
            Node(build_placement_subgraph, mode, mosaic, stacks['c'], filter,
                 TileIndices((2, 0)), viewbox, None)
        ],
    })
    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_two_by_three(self) -> None:
    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None
    masks = None

    mosaic = Mosaic(name='two_by_three',
                    content_hash='content_abc123',
                    stable_hash='mosaic_abc123',
                    target_ids=['a', 'b', 'c', 'd', 'e', 'f'],
                    strips=[['a', 'b', 'c'], ['d', 'e', 'f']],
                    oldest_date=_OLDEST_DATE,
                    newest_date=_NEWEST_DATE,
                    center=[0.0, 0.0],
                    image_scale=1.0,
                    rotation_angle=0.0,
                    footprint=_FOOTPRINT,
                    projected_footprints={})

    context.targets = {
        'a': make_test_target('a', stack_ids=['a']),
        'b': make_test_target('b', stack_ids=['b']),
        'c': make_test_target('c', stack_ids=['c']),
        'd': make_test_target('d', stack_ids=['d']),
        'e': make_test_target('e', stack_ids=['e']),
        'f': make_test_target('f', stack_ids=['f'])
    }

    context.stacks = stacks = {
        'a': make_composite_stack(stable_hash='a'),
        'b': make_composite_stack(stable_hash='b'),
        'c': make_composite_stack(stable_hash='c'),
        'd': make_composite_stack(stable_hash='d'),
        'e': make_composite_stack(stable_hash='e'),
        'f': make_composite_stack(stable_hash='f')
    }

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    place_a = Node(build_placement_subgraph, mode, mosaic, stacks['a'], filter,
                   TileIndices((0, 0)), viewbox, None)
    place_b = Node(build_placement_subgraph, mode, mosaic, stacks['b'], filter,
                   TileIndices((0, 1)), viewbox, None)
    place_c = Node(build_placement_subgraph, mode, mosaic, stacks['c'], filter,
                   TileIndices((0, 2)), viewbox, None)
    place_d = Node(build_placement_subgraph, mode, mosaic, stacks['d'], filter,
                   TileIndices((1, 0)), viewbox, None)
    place_e = Node(build_placement_subgraph, mode, mosaic, stacks['e'], filter,
                   TileIndices((1, 1)), viewbox, None)
    place_f = Node(build_placement_subgraph, mode, mosaic, stacks['f'], filter,
                   TileIndices((1, 2)), viewbox, None)

    merge_a_b = Node(merge_images, mode, mosaic, name='merge_mosaic_abc123_b')
    merge_a_b_c = Node(merge_images, mode, mosaic, name='merge_mosaic_abc123_c')
    merge_d_e = Node(merge_images, mode, mosaic, name='merge_mosaic_abc123_e')
    merge_d_e_f = Node(merge_images, mode, mosaic, name='merge_mosaic_abc123_f')
    final_merge = Node(merge_images,
                       mode,
                       mosaic,
                       name='merge_mosaic_abc123_strip_1')

    expected_graph = Graph({
        place_b: [place_a],
        merge_a_b: [place_a, place_b],
        place_c: [merge_a_b],
        merge_a_b_c: [merge_a_b, place_c],
        place_e: [place_d],
        merge_d_e: [place_d, place_e],
        place_f: [merge_d_e],
        merge_d_e_f: [merge_d_e, place_f],
        final_merge: [merge_a_b_c, merge_d_e_f]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_three_panes(self) -> None:
    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None
    masks = None

    mosaic = Mosaic(name='three_panes',
                    content_hash='content_abc123',
                    stable_hash='mosaic_abc123',
                    target_ids=['a', 'b', 'c'],
                    strips=[['a'], ['b', 'c']],
                    oldest_date=_OLDEST_DATE,
                    newest_date=_NEWEST_DATE,
                    center=[0.0, 0.0],
                    image_scale=1.0,
                    rotation_angle=0.0,
                    footprint=_FOOTPRINT,
                    projected_footprints={})

    context.targets = {
        'a': make_test_target('a', stack_ids=['a']),
        'b': make_test_target('b', stack_ids=['b']),
        'c': make_test_target('c', stack_ids=['c']),
    }

    context.stacks = stacks = {
        'a': make_composite_stack(stable_hash='a'),
        'b': make_composite_stack(stable_hash='b'),
        'c': make_composite_stack(stable_hash='c'),
    }

    _, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    place_a = Node(build_placement_subgraph, mode, mosaic, stacks['a'], filter,
                   TileIndices((0, 0)), viewbox, None)
    place_b = Node(build_placement_subgraph, mode, mosaic, stacks['b'], filter,
                   TileIndices((1, 0)), viewbox, None)
    place_c = Node(build_placement_subgraph, mode, mosaic, stacks['c'], filter,
                   TileIndices((1, 1)), viewbox, None)
    merge_strip_0 = Node(merge_images,
                         mode,
                         mosaic,
                         name='merge_mosaic_abc123_c')
    merge_strip_1 = Node(merge_images,
                         mode,
                         mosaic,
                         name='merge_mosaic_abc123_strip_1')

    expected_graph = Graph({
        place_c: [place_b],
        merge_strip_0: [place_b, place_c],
        merge_strip_1: [
            place_a,
            merge_strip_0,
        ]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_orion_mosaic(self) -> None:
    add_orion_mosaic_images()

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph(regroup=True,
                                make_previews=True,
                                run_main_processing_graph=True)
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    self.assertEqual(len(targets), 4)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    self.assertEqual(mosaic.strips, [['e1f7d909ac125ebe', '3871323c8756dead'],
                                     ['3ef21a508ef46b36', '9ed89ccd9f10cc09']])

    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None
    masks = None

    _, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    place_0_0 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['e1f7d909ac125ebe'], filter,
                     TileIndices((0, 0)), viewbox, None)
    place_0_1 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['3871323c8756dead'], filter,
                     TileIndices((0, 1)), viewbox, None)
    place_1_0 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['3ef21a508ef46b36'], filter,
                     TileIndices((1, 0)), viewbox, None)
    place_1_1 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['9ed89ccd9f10cc09'], filter,
                     TileIndices((1, 1)), viewbox, None)

    merge_0_0__0_1 = Node(merge_images,
                          mode,
                          mosaic,
                          name=f'merge_{mosaic.stable_hash}_3871323c8756dead')
    merge_1_0__1_1 = Node(merge_images,
                          mode,
                          mosaic,
                          name=f'merge_{mosaic.stable_hash}_9ed89ccd9f10cc09')
    final_merge = Node(merge_images,
                       mode,
                       mosaic,
                       name=f'merge_{mosaic.stable_hash}_strip_1')

    expected_graph = Graph({
        place_0_1: [place_0_0],
        merge_0_0__0_1: [place_0_0, place_0_1],
        place_1_1: [place_1_0],
        merge_1_0__1_1: [place_1_0, place_1_1],
        final_merge: [merge_0_0__0_1, merge_1_0__1_1],
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_orion_mosaic_one_by_two(self) -> None:
    add_orion_mosaic_images(panes=[1, 2])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph(regroup=True,
                                make_previews=True,
                                run_main_processing_graph=True)
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    self.assertEqual(len(targets), 2)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))
    print('### mosaic', mosaic)

    self.assertEqual(mosaic.strips,
                     [['e1f7d909ac125ebe'], ['3ef21a508ef46b36']])

    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None
    masks = None

    log, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    expected_graph = Graph({
        Node(merge_images,
             mode,
             mosaic,
             name=f'merge_{mosaic.stable_hash}_strip_1'): [
            Node(build_placement_subgraph,
                 mode, mosaic, stacks['e1f7d909ac125ebe'], filter,
                 TileIndices((0, 0)), viewbox, None),
            Node(build_placement_subgraph,
                 mode, mosaic, stacks['3ef21a508ef46b36'], filter,
                 TileIndices((1, 0)), viewbox, None),
        ]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_orion_mosaic_two_by_one(self) -> None:
    add_orion_mosaic_images(panes=[1, 3])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph(regroup=True,
                                make_previews=True,
                                run_main_processing_graph=True)
    run_graph(graph)

    stacks = context.stacks
    targets = context.targets
    self.assertEqual(len(targets), 2)
    self.assertEqual(len(context.mosaics), 1)
    mosaic = next(iter(context.mosaics.values()))

    self.assertEqual(mosaic.strips, [['e1f7d909ac125ebe', '3871323c8756dead']])

    mode = QualityLevel.SUBFRAME
    filter = 'H'
    viewbox = None
    masks = None

    _, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    place_0_0 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['e1f7d909ac125ebe'], filter,
                     TileIndices((0, 0)), viewbox, None)
    place_0_1 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['3871323c8756dead'], filter,
                     TileIndices((0, 1)), viewbox, None)
    merge = Node(merge_images,
                 mode,
                 mosaic,
                 name=f'merge_{mosaic.stable_hash}_3871323c8756dead')

    expected_graph = Graph({
        place_0_1: [place_0_0],
        merge: [place_0_0, place_0_1]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))

  def test_orion_mosaic_missing_panel(self) -> None:
    add_orion_mosaic_images(panes=[1, 2, 3])

    # Running the graph the first time, with an empty shopping cart, should
    # produce all the subframe previews for the stacks, targets, and mosaic.
    graph = build_compute_graph(regroup=True,
                                make_previews=True,
                                run_main_processing_graph=True)
    run_graph(graph)

    stacks = context.stacks
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
    masks = None

    _, graph = ray.get(
        build_mosaic_filter_subgraph.remote(mode, mosaic, filter, viewbox,
                                            masks))
    print_graph(graph)

    place_0_0 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['3ef21a508ef46b36'], filter,
                     TileIndices((0, 0)), viewbox, None)
    place_1_0 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['e1f7d909ac125ebe'], filter,
                     TileIndices((1, 0)), viewbox, None)
    place_1_1 = Node(build_placement_subgraph, mode,
                     mosaic, stacks['3871323c8756dead'], filter,
                     TileIndices((1, 1)), viewbox, None)
    merge_1_0__1_1 = Node(merge_images,
                          mode,
                          mosaic,
                          name=f'merge_{mosaic.stable_hash}_3871323c8756dead')
    final_merge = Node(merge_images,
                       mode,
                       mosaic,
                       name=f'merge_{mosaic.stable_hash}_strip_1')

    expected_graph = Graph({
        place_1_1: [place_1_0],
        merge_1_0__1_1: [place_1_0, place_1_1],
        final_merge: [place_0_0, merge_1_0__1_1]
    })

    self.assertEqual(canonicalize_graph(graph),
                     canonicalize_graph(expected_graph))
