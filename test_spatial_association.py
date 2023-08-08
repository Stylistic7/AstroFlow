import unittest
import tempfile
import datetime
import numpy as np

import file_io
import make_test_frames
import spatial_association


def _add_dither(ra, dec):
  distance = np.random.uniform(0.0, 0.01)  # degrees
  angle = np.random.uniform(0.0, 2.0 * np.pi)

  return ra + distance * np.sin(distance), dec + np.cos(distance)


class TestSpatialAssociation(unittest.TestCase):
  def test_one_perfect_group(self):
    ids = ['foo'] * 10
    positions = [(300, 40)] * 10

    groups = spatial_association.find_components(ids, positions)
    expected_groups = [('IC1318', ['foo'] * 10)]
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_one_hour_ra(self):
    ids = ['foo'] * 10
    positions = [(300, 40)] * 10

    ids += ['bar'] * 10
    positions += [(315, 40)] * 10

    groups = spatial_association.find_components(ids, positions)
    expected_groups = [('IC1318_RA_300.0_DEC_40.0', ['foo'] * 10),
                       ('IC1318_RA_315.0_DEC_40.0', ['bar'] * 10)]
    self.assertCountEqual(groups, expected_groups)

  def test_two_perfect_groups_one_degree_dec(self):
    ids = ['foo'] * 10
    positions = [(300, 39)] * 10

    ids += ['bar'] * 10
    positions += [(300, 40)] * 10

    groups = spatial_association.find_components(ids, positions)
    expected_groups = [('IC1318_RA_300.0_DEC_39.0', ['foo'] * 10),
                       ('IC1318_RA_300.0_DEC_40.0', ['bar'] * 10)]
    self.assertCountEqual(groups, expected_groups)

  def test_two_perfect_groups_half_degree_dec(self):
    ids = ['foo'] * 10
    positions = [(300, 39.5)] * 10

    ids += ['bar'] * 10
    positions += [(300, 40)] * 10

    groups = spatial_association.find_components(ids, positions)
    expected_groups = [('IC1318_RA_300.0_DEC_39.5', ['foo'] * 10),
                       ('IC1318_RA_300.0_DEC_40.0', ['bar'] * 10)]
    self.assertCountEqual(groups, expected_groups)

  def test_two_noisy_groups(self):
    ids = ['foo'] * 10
    positions = [_add_dither(307, 40) for i in range(10)]

    ids += ['bar'] * 10
    positions += [_add_dither(315, 45) for i in range(10)]

    groups = spatial_association.find_components(ids, positions)
    expected_groups = [('North America nebula', ['bar'] * 10),
                       ('IC1318', ['foo'] * 10)]
    self.assertCountEqual(groups, expected_groups)

  def test_sadr_pano(self):
    print('Entering test_sadr_pano')

    positions = [
        (301.625, 41.37361111111111),
        (301.81666666666666, 38.29111111111111),
        (304.34583333333336, 41.44222222222222),
        (304.4125, 38.35472222222222),
        (307.06666666666666, 41.44638888888888),
        (307.01666666666665, 38.36083333333333),
    ]
    ids = ['Pane 1', 'Pane 2', 'Pane 3', 'Pane 4', 'Pane 5', 'Pane 6']

    groups = spatial_association.find_components(ids,
                                                 positions,
                                                 min_component_members=1)

    expected_groups = [
        ('IC1311', ['Pane 1']),
        ('IC1318_RA_301.8_DEC_38.3', ['Pane 2']),
        ('IC1318_RA_304.3_DEC_41.4', ['Pane 3']),
        ('IC1318_RA_304.4_DEC_38.4', ['Pane 4']),
        ('IC1318_RA_307.1_DEC_41.4', ['Pane 5']),
        ('IC1318_RA_307.0_DEC_38.4', ['Pane 6']),
    ]
    self.assertCountEqual(groups, expected_groups)
