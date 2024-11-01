import unittest
from ray_infra import ray_local

from dataclasses import asdict
import datetime
import ray
from context import context, get_cache
import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from stacks.types import StackHash
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack
from targets import Target
from targets.types import TargetHash
from mosaics import Mosaic
from testing.test_context import make_test_context
from testing.test_stacks import make_composite_stack
from testing.testdata_cache import copy_testdata_into_cache

from mosaics import identification


class TestMosaicIdentification(unittest.TestCase):

  def setUp(self) -> None:
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def test_two_targets(self) -> None:
    # These two targets are on the celestial equator, but overlap by one degree
    # in right ascension. This is the easiest case to think about.
    target0 = Target(name='foo',
                     stable_hash=TargetHash('geometric_foo'),
                     content_hash='content_foo',
                     oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
                     newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
                     stack_ids=[StackHash('0'),
                                StackHash('1'),
                                StackHash('2')],
                     center=[100, 50],
                     footprint=SingleSphericalPolygon.from_radec(
                         [98, 102, 102, 98], [-1, -1, 1, 1]),
                     projected_footprints={})
    target1 = Target(name='bar',
                     stable_hash=TargetHash('geometric_bar'),
                     content_hash='content_bar',
                     oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
                     newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
                     stack_ids=[StackHash('3'),
                                StackHash('4'),
                                StackHash('5')],
                     center=[100, 51],
                     footprint=SingleSphericalPolygon.from_radec(
                         [95, 99, 99, 95], [-1, -1, 1, 1]),
                     projected_footprints={})

    context.grouped_stacks = {
        str(i): make_composite_stack(stable_hash=str(i), image_scale=1)
        for i in range(6)
    }
    context.grouped_targets = {
        target0.stable_hash: target0,
        target1.stable_hash: target1
    }

    ray.get(identification.identify_mosaics.remote())
    mosaics = context.grouped_mosaics
    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))

    self.assertEqual(mosaic.target_ids,
                     [target0.stable_hash, target1.stable_hash])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 1)])

    # TODO: This only works when the mosaic polygon is "tight," rather than
    # the convex hull.
    # The mosaic is 7x2, while the two panels are each 4x2.
    # target_areas = [t.footprint.area() for t in targets.values()]
    # ratio = mosaic.footprint.area() / sum(target_areas)
    # self.assertTrue(np.isclose(ratio, (7 * 2) / (4 * 2 + 4 * 2), atol=1e-3))

  def test_three_targets(self) -> None:
    foo = Target(name='foo',
                 stable_hash=TargetHash('geometric_foo'),
                 content_hash='content_foo',
                 oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
                 stack_ids=[StackHash('0'),
                            StackHash('1'),
                            StackHash('2')],
                 center=[100, 0],
                 footprint=SingleSphericalPolygon.from_radec([98, 102, 102, 98],
                                                             [-1, -1, 1, 1]),
                 projected_footprints={})
    bar = Target(name='bar',
                 stable_hash=TargetHash('geometric_bar'),
                 content_hash='content_bar',
                 oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
                 stack_ids=[StackHash('3'),
                            StackHash('4'),
                            StackHash('5')],
                 center=[97, 0],
                 footprint=SingleSphericalPolygon.from_radec([95, 99, 99, 95],
                                                             [-1, -1, 1, 1]),
                 projected_footprints={})
    baz = Target(name='baz',
                 stable_hash=TargetHash('geometric_baz'),
                 content_hash='content_baz',
                 oldest_date=datetime.datetime(2024, 4, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 5, 6, 16, 29, 43),
                 stack_ids=[StackHash('6'),
                            StackHash('7'),
                            StackHash('8')],
                 center=[97, 1],
                 footprint=SingleSphericalPolygon.from_radec([95, 99, 99, 95],
                                                             [0, 0, 2, 2]),
                 projected_footprints={})

    context.grouped_stacks = {
        str(i): make_composite_stack(stable_hash=str(i), image_scale=1)
        for i in range(9)
    }
    context.grouped_targets = {
        foo.stable_hash: foo,
        bar.stable_hash: bar,
        baz.stable_hash: baz
    }

    ray.get(identification.identify_mosaics.remote())
    mosaics = context.grouped_mosaics

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    self.assertEqual(mosaic.target_ids,
                     [bar.stable_hash, baz.stable_hash, foo.stable_hash])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 1), (1, 0)])

    # TODO: This only works when the mosaic polygon is "tight," rather than
    # the convex hull.
    # The mosaic is 7x2 + 4x1, while the three panels are each 4x2.
    # target_areas = [t.footprint.area() for t in targets.values()]
    # ratio = mosaic.footprint.area() / sum(target_areas)
    # self.assertTrue(np.isclose(ratio, (7 * 2 + 4 * 1) / (3 * 4 * 2), atol=1e-3))

  def test_three_out_of_order_targets(self) -> None:
    foo = Target(name='foo',
                 stable_hash=TargetHash('geometric_foo'),
                 content_hash='content_foo',
                 oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
                 stack_ids=[StackHash('0'),
                            StackHash('1'),
                            StackHash('2')],
                 center=[100, 1],
                 footprint=SingleSphericalPolygon.from_radec([98, 102, 102, 98],
                                                             [0, 0, 2, 2]),
                 projected_footprints={})
    bar = Target(name='bar',
                 stable_hash=TargetHash('geometric_bar'),
                 content_hash='content_bar',
                 oldest_date=datetime.datetime(2024, 4, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 5, 6, 16, 29, 43),
                 stack_ids=[StackHash('6'),
                            StackHash('7'),
                            StackHash('8')],
                 center=[100, 3],
                 footprint=SingleSphericalPolygon.from_radec([98, 102, 102, 98],
                                                             [2, 2, 4, 4]),
                 projected_footprints={})
    baz = Target(name='baz',
                 stable_hash=TargetHash('geometric_baz'),
                 content_hash='content_baz',
                 oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
                 stack_ids=[StackHash('3'),
                            StackHash('4'),
                            StackHash('5')],
                 center=[100, 2],
                 footprint=SingleSphericalPolygon.from_radec([98, 102, 102, 98],
                                                             [1, 1, 3, 3]),
                 projected_footprints={})

    context.grouped_stacks = {
        str(i): make_composite_stack(stable_hash=str(i), image_scale=1)
        for i in range(9)
    }
    context.grouped_targets = {
        foo.stable_hash: foo,
        bar.stable_hash: bar,
        baz.stable_hash: baz
    }

    ray.get(identification.identify_mosaics.remote())
    mosaics = context.grouped_mosaics

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    self.assertEqual(mosaic.target_ids,
                     [foo.stable_hash, bar.stable_hash, baz.stable_hash])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 2), (0, 1)])

  def test_three_targets_and_an_outlier(self) -> None:
    foo = Target(name='foo',
                 stable_hash=TargetHash('geometric_foo'),
                 content_hash='content_foo',
                 oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
                 stack_ids=[StackHash('0'),
                            StackHash('1'),
                            StackHash('2')],
                 center=[100, 0],
                 footprint=SingleSphericalPolygon.from_radec([98, 102, 102, 98],
                                                             [-1, -1, 1, 1]),
                 projected_footprints={})
    bar = Target(name='bar',
                 stable_hash=TargetHash('geometric_bar'),
                 content_hash='content_bar',
                 oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
                 stack_ids=[StackHash('3'),
                            StackHash('4'),
                            StackHash('5')],
                 center=[97, 0],
                 footprint=SingleSphericalPolygon.from_radec([95, 99, 99, 95],
                                                             [-1, -1, 1, 1]),
                 projected_footprints={})
    baz = Target(name='baz',
                 stable_hash=TargetHash('geometric_baz'),
                 content_hash='content_baz',
                 oldest_date=datetime.datetime(2024, 4, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 5, 6, 16, 29, 43),
                 stack_ids=[StackHash('6'),
                            StackHash('7'),
                            StackHash('8')],
                 center=[97, 1],
                 footprint=SingleSphericalPolygon.from_radec([95, 99, 99, 95],
                                                             [0, 0, 2, 2]),
                 projected_footprints={})
    outlier = Target(
        name='outlier',
        stable_hash=TargetHash('geometric_outlier'),
        content_hash='content_outlier',
        oldest_date=datetime.datetime(2024, 6, 6, 16, 29, 43),
        newest_date=datetime.datetime(2024, 7, 6, 16, 29, 43),
        stack_ids=[StackHash('9'),
                   StackHash('10'),
                   StackHash('11')],
        center=[98, -0.5],
        footprint=SingleSphericalPolygon.from_radec([96, 100, 100, 96],
                                                    [-1.5, -1.5, 0.5, 0.5]),
        projected_footprints={})

    context.grouped_stacks = {
        str(i): make_composite_stack(stable_hash=str(i), image_scale=1)
        for i in range(12)
    }
    context.grouped_targets = {
        foo.stable_hash: foo,
        bar.stable_hash: bar,
        baz.stable_hash: baz,
        outlier.stable_hash: outlier,
    }

    ray.get(identification.identify_mosaics.remote())
    mosaics = context.grouped_mosaics

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    self.assertEqual(mosaic.target_ids,
                     [bar.stable_hash, baz.stable_hash, foo.stable_hash])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 1), (1, 0)])

  def test_ic1318(self) -> None:
    foo = Target(name='foo',
                 stable_hash=TargetHash('geometric_foo'),
                 content_hash='content_foo',
                 oldest_date=datetime.datetime(2023, 12, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 1, 6, 16, 29, 43),
                 stack_ids=[StackHash('1'),
                            StackHash('2'),
                            StackHash('3')],
                 center=[100, 0],
                 footprint=SingleSphericalPolygon(
                     np.array([[0.45603681, -0.56689475, 0.6860472],
                               [0.48206283, -0.60210192, 0.6364658],
                               [0.44676643, -0.62853698, 0.63666398],
                               [0.420705, -0.59334582, 0.68625654],
                               [0.45603681, -0.56689475, 0.6860472]]),
                     np.array([0.45175641, -0.59818691, 0.66188259])),
                 projected_footprints={})

    bar = Target(name='bar',
                 stable_hash=TargetHash('geometric_bar'),
                 content_hash='content_bar',
                 oldest_date=datetime.datetime(2024, 2, 6, 16, 29, 43),
                 newest_date=datetime.datetime(2024, 3, 6, 16, 29, 43),
                 stack_ids=[StackHash('3'),
                            StackHash('4'),
                            StackHash('5')],
                 center=[97, 0],
                 footprint=SingleSphericalPolygon(
                     np.array([[0.47687137, -0.5960087, 0.64603973],
                               [0.50124499, -0.62906976, 0.59415881],
                               [0.46588868, -0.65547604, 0.59438951],
                               [0.44152953, -0.62243363, 0.64624148],
                               [0.47687137, -0.5960087, 0.64603973]]),
                     np.array([0.47176113, -0.62623935, 0.62069777])),
                 projected_footprints={})

    context.grouped_stacks = {
        str(i): make_composite_stack(stable_hash=str(i), image_scale=1)
        for i in range(6)
    }
    context.grouped_targets = {
        foo.stable_hash: foo,
        bar.stable_hash: bar,
    }

    ray.get(identification.identify_mosaics.remote())
    mosaics = context.grouped_mosaics

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    print('### mosaic', mosaic)

    self.assertEqual(mosaic.target_ids, [bar.stable_hash, foo.stable_hash])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 1)])

  def test_big_cygnus_pano(self) -> None:
    cache = get_cache()
    copy_testdata_into_cache()

    ray.get(identification.identify_mosaics.remote(debug=True))
    targets = context.grouped_targets
    mosaics = context.grouped_mosaics

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    print('### mosaic', mosaic)

    target_a = targets['463edef05b214da9']
    target_b = targets['e163f2b56b9a02aa']

    print(f'### target_a', asdict(target_a))

    self.assertFalse(identification.compatible_overlap(target_a, target_b))


#   def test_compatible_overlap(self) -> None:
#     a = SingleSphericalPolygon(
#         np.array([[0.45603681, -0.56689475, 0.6860472],
#                   [0.48206283, -0.60210192, 0.6364658],
#                   [0.44676643, -0.62853698, 0.63666398],
#                   [0.420705, -0.59334582, 0.68625654],
#                   [0.45603681, -0.56689475, 0.6860472]]),
#         np.array([0.45175641, -0.59818691, 0.66188259]))

#     b = SingleSphericalPolygon(
#         np.array([[0.47687137, -0.5960087, 0.64603973],
#                   [0.50124499, -0.62906976, 0.59415881],
#                   [0.46588868, -0.65547604, 0.59438951],
#                   [0.44152953, -0.62243363, 0.64624148],
#                   [0.47687137, -0.5960087, 0.64603973]]),
#         np.array([0.47176113, -0.62623935, 0.62069777]))

#     self.assertFalse(identification.compatible_overlap(a, b))

if __name__ == "__main__":
  unittest.main()
