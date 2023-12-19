import unittest

import shapely

from target import Target
from mosaic import Mosaic

from . import identification


class TestMosaicIdentification(unittest.TestCase):
  def test_one_target(self):
    targets = {
        'foo':
        Target(id='foo',
               name='foo',
               stack_ids=[1, 2, 3],
               center=(100, 50),
               footprint=[(98, 49), (98, 51), (102, 51), (102, 49)])
    }

    mosaics = identification.identify_mosiacs(targets)

    self.assertEqual(len(mosaics), 1)
    self.assertEqual(mosaics[0].target_ids, ['foo'])
    self.assertEqual(mosaics[0].tile_indices, [(0, 0)])
    self.assertTrue(
        shapely.Polygon(mosaics[0].footprint).equals(
            shapely.Polygon([(98, 49), (98, 51), (102, 51), (102, 49)])))
    self.assertEqual(mosaics[0].viewbox, [98.0, 49.0, 4.0, 2.0])

  def test_two_targets(self):
    targets = {
        'foo':
        Target(id='foo',
               name='foo',
               stack_ids=[1, 2, 3],
               center=(100, 50),
               footprint=[(98, 49), (102, 49), (102, 51), (98, 51)]),
        'bar':
        Target(id='bar',
               name='bar',
               stack_ids=[1, 2, 3],
               center=(100, 51),
               footprint=[(98, 50), (102, 50), (102, 52), (98, 52)])
    }

    mosaics = identification.identify_mosiacs(targets)

    self.assertEqual(len(mosaics), 1)
    self.assertEqual(mosaics[0].target_ids, ['foo', 'bar'])
    self.assertEqual(mosaics[0].tile_indices, [(0, 0), (0, 1)])
    self.assertTrue(
        shapely.Polygon(mosaics[0].footprint).equals(
            shapely.Polygon([(98, 49), (98, 52), (102, 52), (102, 49)])))
    self.assertEqual(mosaics[0].viewbox, [98.0, 49.0, 4.0, 3.0])

  def test_three_targets(self):
    targets = {
        'foo':
        Target(id='foo',
               name='foo',
               stack_ids=[1, 2, 3],
               center=(100, 50),
               footprint=[(98, 49), (102, 49), (102, 51), (98, 51)]),
        'bar':
        Target(id='bar',
               name='bar',
               stack_ids=[1, 2, 3],
               center=(100, 51),
               footprint=[(98, 50), (102, 50), (102, 52), (98, 52)]),
        'baz':
        Target(id='baz',
               name='baz',
               stack_ids=[1, 2, 3],
               center=(102, 50),
               footprint=[(100, 49), (104, 49), (104, 51), (100, 51)])
    }

    mosaics = identification.identify_mosiacs(targets)

    self.assertEqual(len(mosaics), 1)
    self.assertEqual(mosaics[0].target_ids, ['foo', 'bar', 'baz'])
    self.assertEqual(mosaics[0].tile_indices, [(0, 0), (0, 1), (1, 0)])
    self.assertTrue(
        shapely.Polygon(mosaics[0].footprint).equals(
            shapely.Polygon([(98, 49), (104, 49), (104, 51), (102, 51),
                             (102, 52), (98, 52), (98, 49)])))
    self.assertEqual(mosaics[0].viewbox, [98.0, 49.0, 6.0, 3.0])


if __name__ == "__main__":
  unittest.main()