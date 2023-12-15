import unittest
import tempfile
import datetime
import numpy as np
import random

import file_io
from make_test_frames import make_test_frames
import spatial_association
from plate_solving import PlateSolution
from cache import cache
import datetime


class TestSpatialAssociation(unittest.TestCase):
  def test_one_perfect_group(self):
    fits_headers, plate_solutions = make_test_frames(ra=100,
                                                     dec=50,
                                                     prefix='foo',
                                                     number=10,
                                                     dither=0)

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames, plate_solutions,
                                                 fits_headers)

    expected_groups = {'IC360': list(sorted(plate_solutions.keys()))}
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_one_hour_ra(self):
    f_foo, p_foo = make_test_frames(ra=300,
                                    dec=40,
                                    prefix='foo',
                                    number=10,
                                    dither=0)
    f_bar, p_bar = make_test_frames(ra=315,
                                    dec=40,
                                    prefix='bar',
                                    number=10,
                                    dither=0)
    fits_headers = f_foo | f_bar
    plate_solutions = p_foo | p_bar

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames, plate_solutions,
                                                 fits_headers)

    expected_groups = ['IC1318_RA_300.0_DEC_40.0', 'IC1318_RA_315.0_DEC_40.0']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(
        all(f.startswith('foo') for f in groups['IC1318_RA_300.0_DEC_40.0']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['IC1318_RA_315.0_DEC_40.0']))

  def test_two_perfect_groups_one_degree_dec(self):
    f_foo, p_foo = make_test_frames(ra=300,
                                    dec=39,
                                    prefix='foo',
                                    number=10,
                                    dither=0)
    f_bar, p_bar = make_test_frames(ra=300,
                                    dec=40,
                                    prefix='bar',
                                    number=10,
                                    dither=0)
    fits_headers = f_foo | f_bar
    plate_solutions = p_foo | p_bar

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames, plate_solutions,
                                                 fits_headers)

    expected_groups = ['IC1318_RA_300.0_DEC_39.0', 'IC1318_RA_300.0_DEC_40.0']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(
        all(f.startswith('foo') for f in groups['IC1318_RA_300.0_DEC_39.0']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['IC1318_RA_300.0_DEC_40.0']))

  def test_two_perfect_groups_half_degree_dec(self):
    f_foo, p_foo = make_test_frames(ra=300,
                                    dec=39.5,
                                    prefix='foo',
                                    number=10,
                                    dither=0,
                                    rotation_angle_deg=0)
    f_bar, p_bar = make_test_frames(ra=300,
                                    dec=40,
                                    prefix='bar',
                                    number=10,
                                    dither=0,
                                    rotation_angle_deg=0)
    fits_headers = f_foo | f_bar
    plate_solutions = p_foo | p_bar

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames, plate_solutions,
                                                 fits_headers)

    expected_groups = ['IC1318_RA_300.0_DEC_39.5', 'IC1318_RA_300.0_DEC_40.0']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(
        all(f.startswith('foo') for f in groups['IC1318_RA_300.0_DEC_39.5']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['IC1318_RA_300.0_DEC_40.0']))

  def test_two_noisy_groups(self):
    f_foo, p_foo = make_test_frames(ra=307, dec=40, prefix='foo', number=10)
    f_bar, p_bar = make_test_frames(ra=315, dec=45, prefix='bar', number=10)
    fits_headers = f_foo | f_bar
    plate_solutions = p_foo | p_bar

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames, plate_solutions,
                                                 fits_headers)

    expected_groups = ['North America nebula', 'IC1318']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(all(f.startswith('foo') for f in groups['IC1318']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['North America nebula']))

  def test_sadr_pano(self):
    print('Entering test_sadr_pano')

    centers = [
        (301.625, 41.37361111111111),
        (301.81666666666666, 38.29111111111111),
        (304.34583333333336, 41.44222222222222),
        (304.4125, 38.35472222222222),
        (307.06666666666666, 41.44638888888888),
        (307.01666666666665, 38.36083333333333),
    ]
    ids = ['Pane 0', 'Pane 1', 'Pane 2', 'Pane 3', 'Pane 4', 'Pane 5']

    fits_headers, plate_solutions = make_test_frames(centers=centers,
                                                     ids=ids,
                                                     dither=0)

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames,
                                                 plate_solutions,
                                                 fits_headers,
                                                 min_component_members=1)
    expected_groups = {
        'IC1311': ['Pane 0'],
        'IC1318_RA_301.8_DEC_38.3': ['Pane 1'],
        'IC1318_RA_304.3_DEC_41.4': ['Pane 2'],
        'IC1318_RA_304.4_DEC_38.4': ['Pane 3'],
        'IC1318_RA_307.1_DEC_41.4': ['Pane 4'],
        'IC1318_RA_307.0_DEC_38.4': ['Pane 5'],
    }
    self.assertCountEqual(groups.keys(), expected_groups)
