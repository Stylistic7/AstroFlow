import unittest

import numpy as np
from spherical_geometry.polygon import SphericalPolygon

from target import Target
from mosaic import Mosaic

from . import identification


class TestMosaicIdentification(unittest.TestCase):

  def test_two_targets(self):
    # These two targets are on the celestial equator, but overlap by one degree
    # in right ascension. This is the easiest case to think about.
    targets = {
        'foo':
            Target(id='foo',
                   name='foo',
                   stack_ids=[1, 2, 3],
                   center=(100, 50),
                   footprint=SphericalPolygon.from_radec([98, 102, 102, 98],
                                                         [-1, -1, 1, 1])),
        'bar':
            Target(id='bar',
                   name='bar',
                   stack_ids=[1, 2, 3],
                   center=(100, 51),
                   footprint=SphericalPolygon.from_radec([95, 99, 99, 95],
                                                         [-1, -1, 1, 1]))
    }

    mosaics = identification.identify_mosiacs(targets)

    self.assertEqual(len(mosaics), 1)
    self.assertEqual(mosaics[0].target_ids, ['foo', 'bar'])
    self.assertEqual(mosaics[0].tile_indices, [(0, 0), (0, 1)])

    # The mosaic is 7x2, while the two panels are each 4x2.
    target_areas = [t.footprint_polygon().area() for t in targets.values()]
    ratio = mosaics[0].footprint_polygon().area() / sum(target_areas)
    self.assertTrue(np.isclose(ratio, (7 * 2) / (4 * 2 + 4 * 2), atol=1e-3))

  def test_three_targets(self):
    targets = {
        'foo':
            Target(id='foo',
                   name='foo',
                   stack_ids=[1, 2, 3],
                   center=(100, 50),
                   footprint=SphericalPolygon.from_radec([98, 102, 102, 98],
                                                         [-1, -1, 1, 1])),
        'bar':
            Target(id='bar',
                   name='bar',
                   stack_ids=[1, 2, 3],
                   center=(100, 51),
                   footprint=SphericalPolygon.from_radec([95, 99, 99, 95],
                                                         [-1, -1, 1, 1])),
        'baz':
            Target(id='baz',
                   name='baz',
                   stack_ids=[1, 2, 3],
                   center=(102, 50),
                   footprint=SphericalPolygon.from_radec([95, 99, 99, 95],
                                                         [0, 0, 2, 2]))
    }

    mosaics = identification.identify_mosiacs(targets)

    self.assertEqual(len(mosaics), 1)
    self.assertEqual(mosaics[0].target_ids, ['foo', 'bar', 'baz'])
    self.assertEqual(mosaics[0].tile_indices, [(0, 0), (0, 1), (1, 0)])

    # The mosaic is 7x2 + 4x1, while the three panels are each 4x2.
    target_areas = [t.footprint_polygon().area() for t in targets.values()]
    ratio = mosaics[0].footprint_polygon().area() / sum(target_areas)
    self.assertTrue(np.isclose(ratio, (7 * 2 + 4 * 1) / (3 * 4 * 2), atol=1e-3))


if __name__ == "__main__":
  unittest.main()
