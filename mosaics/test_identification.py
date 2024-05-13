import unittest
import ray_local

import ray
import numpy as np
from spherical_geometry.polygon import SphericalPolygon

from target import Target
from mosaic import Mosaic
from make_test_context import make_test_context

from . import identification


class TestMosaicIdentification(unittest.TestCase):

  def setUp(self):
    self.context = make_test_context()

  def test_two_targets(self):
    # These two targets are on the celestial equator, but overlap by one degree
    # in right ascension. This is the easiest case to think about.
    target0 = Target(name='foo',
                     stack_ids=[1, 2, 3],
                     center=(100, 50),
                     footprint=SphericalPolygon.from_radec([98, 102, 102, 98],
                                                           [-1, -1, 1, 1]))
    target1 = Target(name='bar',
                     stack_ids=[4, 5, 6],
                     center=(100, 51),
                     footprint=SphericalPolygon.from_radec([95, 99, 99, 95],
                                                           [-1, -1, 1, 1]))

    targets = {target0.hash(): target0, target1.hash(): target1}
    self.context.targets = ray.put(targets)
    self.context, _ = ray.get(
        identification.identify_mosiacs.remote(self.context))
    mosaics = ray.get(self.context.mosaics)
    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))

    self.assertEqual(mosaic.target_ids, [target0.hash(), target1.hash()])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 1)])

    # The mosaic is 7x2, while the two panels are each 4x2.
    target_areas = [t.footprint_polygon().area() for t in targets.values()]
    ratio = mosaic.footprint_polygon().area() / sum(target_areas)
    self.assertTrue(np.isclose(ratio, (7 * 2) / (4 * 2 + 4 * 2), atol=1e-3))

  def test_three_targets(self):
    target0 = Target(name='foo',
                     stack_ids=[1, 2, 3],
                     center=(100, 50),
                     footprint=SphericalPolygon.from_radec([98, 102, 102, 98],
                                                           [-1, -1, 1, 1]))
    target1 = Target(name='bar',
                     stack_ids=[4, 5, 6],
                     center=(100, 51),
                     footprint=SphericalPolygon.from_radec([95, 99, 99, 95],
                                                           [-1, -1, 1, 1]))
    target2 = Target(name='baz',
                     stack_ids=[7, 8, 9],
                     center=(102, 50),
                     footprint=SphericalPolygon.from_radec([95, 99, 99, 95],
                                                           [0, 0, 2, 2]))

    targets = {
        target0.hash(): target0,
        target1.hash(): target1,
        target2.hash(): target2
    }

    self.context.targets = ray.put(targets)
    self.context, _ = ray.get(
        identification.identify_mosiacs.remote(self.context))
    mosaics = ray.get(self.context.mosaics)

    self.assertEqual(len(mosaics), 1)
    mosaic = next(iter(mosaics.values()))
    self.assertEqual(
        mosaic.target_ids,
        [target0.hash(), target1.hash(),
         target2.hash()])
    self.assertEqual(mosaic.tile_indices, [(0, 0), (0, 1), (1, 0)])

    # The mosaic is 7x2 + 4x1, while the three panels are each 4x2.
    target_areas = [t.footprint_polygon().area() for t in targets.values()]
    ratio = mosaic.footprint_polygon().area() / sum(target_areas)
    self.assertTrue(np.isclose(ratio, (7 * 2 + 4 * 1) / (3 * 4 * 2), atol=1e-3))


if __name__ == "__main__":
  unittest.main()
