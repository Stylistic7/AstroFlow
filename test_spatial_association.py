import unittest
import tempfile
from datetime import datetime, timedelta
import numpy as np
import random

import file_io
from make_test_frames import make_test_frames
import spatial_association
from plate_solving import PlateSolution
from cache import cache


class TestSpatialAssociation(unittest.TestCase):
  def test_one_perfect_group(self):
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        ra=100, dec=50, prefix='foo', number=10, dither=0)

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames, plate_solutions,
                                                 fits_headers)

    expected_groups = [('IC360', list(sorted(image_hashes.keys())))]
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_one_hour_ra(self):
    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime,
        ra=300,
        dec=40,
        prefix='foo',
        number=3,
        dither=0)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=315,
        dec=40,
        prefix='bar',
        number=3,
        dither=0,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    file_for_hash = {v: k for k, v in image_hashes.items()}

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames,
                                                 plate_solutions,
                                                 fits_headers,
                                                 min_component_members=3)

    expected_groups = [('IC1318', [
        file_for_hash['foo_00000.fit'], file_for_hash['foo_00001.fit'],
        file_for_hash['foo_00002.fit']
    ]),
                       ('IC1318', [
                           file_for_hash['bar_00000.fit'],
                           file_for_hash['bar_00001.fit'],
                           file_for_hash['bar_00002.fit']
                       ])]
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_one_degree_dec(self):
    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime,
        ra=300,
        dec=39,
        prefix='foo',
        number=3,
        dither=0)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=300,
        dec=40,
        prefix='bar',
        number=3,
        dither=0,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    file_for_hash = {v: k for k, v in image_hashes.items()}

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames,
                                                 plate_solutions,
                                                 fits_headers,
                                                 min_component_members=3)

    expected_groups = [
        ('IC1318', [
            file_for_hash['foo_00000.fit'], file_for_hash['foo_00001.fit'],
            file_for_hash['foo_00002.fit']
        ]),
        ('IC1318', [
            file_for_hash['bar_00000.fit'], file_for_hash['bar_00001.fit'],
            file_for_hash['bar_00002.fit']
        ]),
    ]
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_half_degree_dec(self):
    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime,
        ra=300,
        dec=39.5,
        prefix='foo',
        number=3,
        dither=0,
        rotation_angle_deg=0)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=300,
        dec=40,
        prefix='bar',
        number=3,
        dither=0,
        rotation_angle_deg=0,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    file_for_hash = {v: k for k, v in image_hashes.items()}

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames,
                                                 plate_solutions,
                                                 fits_headers,
                                                 min_component_members=3)

    expected_groups = [
        ('IC1318', [
            file_for_hash['foo_00000.fit'], file_for_hash['foo_00001.fit'],
            file_for_hash['foo_00002.fit']
        ]),
        ('IC1318', [
            file_for_hash['bar_00000.fit'], file_for_hash['bar_00001.fit'],
            file_for_hash['bar_00002.fit']
        ]),
    ]
    self.assertEqual(groups, expected_groups)

  def test_two_noisy_groups(self):
    start_datetime = datetime(2023, 12, 6, 16, 29, 43)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime, ra=307, dec=40, prefix='foo', number=3)
    image_hashes, fits_headers, plate_solutions = make_test_frames(
        start_datetime=start_datetime + timedelta(hours=1),
        ra=315,
        dec=45,
        prefix='bar',
        number=3,
        image_hashes=image_hashes,
        fits_headers=fits_headers,
        plate_solutions=plate_solutions)

    file_for_hash = {v: k for k, v in image_hashes.items()}

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames,
                                                 plate_solutions,
                                                 fits_headers,
                                                 min_component_members=3)

    expected_groups = [
        ('IC1318', [
            file_for_hash['foo_00000.fit'], file_for_hash['foo_00001.fit'],
            file_for_hash['foo_00002.fit']
        ]),
        ('North America nebula', [
            file_for_hash['bar_00000.fit'], file_for_hash['bar_00001.fit'],
            file_for_hash['bar_00002.fit']
        ]),
    ]
    self.assertEqual(groups, expected_groups)

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

    image_hashes, fits_headers, plate_solutions = make_test_frames(
        centers=centers, ids=ids, dither=0)

    file_for_hash = {v: k for k, v in image_hashes.items()}

    light_frames = list(plate_solutions.keys())
    groups = spatial_association.find_components(light_frames,
                                                 plate_solutions,
                                                 fits_headers,
                                                 min_component_members=1)
    expected_groups = [('IC1311', [file_for_hash['Pane 0']]),
                       ('IC1318', [file_for_hash['Pane 1']]),
                       ('IC1318', [file_for_hash['Pane 2']]),
                       ('IC1318', [file_for_hash['Pane 3']]),
                       ('IC1318', [file_for_hash['Pane 4']]),
                       ('IC1318', [file_for_hash['Pane 5']])]
    self.assertEqual(groups, expected_groups)