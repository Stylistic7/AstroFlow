import unittest
import tempfile
import datetime
import numpy as np

import file_io
import make_test_frames
import spatial_association
from plate_solving import PlateSolution
from cache import cache
import datetime


def _add_dither(center):
  ra, dec = center
  distance = np.random.uniform(0.0, 0.01)  # degrees
  angle = np.random.uniform(0.0, 2.0 * np.pi)

  return ra + distance * np.sin(distance), dec + np.cos(distance)


def footprint_around_center(center, ra_span=4, dec_span=2):
  ra, dec = center
  return [(ra - ra_span / 2, dec + dec_span / 2),
          (ra - ra_span / 2, dec - dec_span / 2),
          (ra + ra_span / 2, dec - dec_span / 2),
          (ra + ra_span / 2, dec + dec_span / 2)]


def make_plate_solutions(center, prefix, number=1, dither=False):
  plate_solutions = {}
  for i in range(number):
    image_filename = f'{prefix}_{i}' if number > 1 else prefix
    plate_solutions[image_filename] = PlateSolution(
        image_filename=image_filename,
        center=_add_dither(center) if dither else center,
        footprint=footprint_around_center(center))

  return plate_solutions


def make_fits_headers(prefix, number=1):
  date_obs = datetime.datetime(2007, 12, 6, 16, 29, 43, 79043)
  fits_headers = {}
  for i in range(number):
    image_filename = f'{prefix}_{i}' if number > 1 else prefix
    fits_headers[image_filename] = {'DATE-OBS': date_obs}
    date_obs += datetime.timedelta(minutes=5)

  return fits_headers


class TestSpatialAssociation(unittest.TestCase):
  def test_one_perfect_group(self):
    footprint = [(98, 51), (98, 49), (102, 49), (102, 51)]

    plate_solutions = {}
    for i in range(10):
      image_filename = f'foo_{i}'
      plate_solutions[image_filename] = PlateSolution(
          image_filename=image_filename, center=(100, 50), footprint=footprint)

    groups = spatial_association.find_components(plate_solutions, fits_headers)

    expected_groups = {'IC360': list(sorted(plate_solutions.keys()))}
    self.assertEqual(groups, expected_groups)

  def test_two_perfect_groups_one_hour_ra(self):
    plate_solutions = make_plate_solutions((300, 40), 'foo', 10)
    fits_headers = make_fits_headers('foo', 10)
    plate_solutions |= make_plate_solutions((315, 40), 'bar', 10)
    fits_headers |= make_fits_headers('bar', 10)

    groups = spatial_association.find_components(plate_solutions, fits_headers)

    expected_groups = ['IC1318_RA_300.0_DEC_40.0', 'IC1318_RA_315.0_DEC_40.0']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(
        all(f.startswith('foo') for f in groups['IC1318_RA_300.0_DEC_40.0']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['IC1318_RA_315.0_DEC_40.0']))

  def test_two_perfect_groups_one_degree_dec(self):
    plate_solutions = make_plate_solutions((300, 39), 'foo', 10)
    fits_headers = make_fits_headers('foo', 10)
    plate_solutions |= make_plate_solutions((300, 40), 'bar', 10)
    fits_headers |= make_fits_headers('bar', 10)

    groups = spatial_association.find_components(plate_solutions, fits_headers)

    expected_groups = ['IC1318_RA_300.0_DEC_39.0', 'IC1318_RA_300.0_DEC_40.0']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(
        all(f.startswith('foo') for f in groups['IC1318_RA_300.0_DEC_39.0']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['IC1318_RA_300.0_DEC_40.0']))

  def test_two_perfect_groups_half_degree_dec(self):
    plate_solutions = make_plate_solutions((300, 39.5), 'foo', 10)
    fits_headers = make_fits_headers('foo', 10)
    plate_solutions |= make_plate_solutions((300, 40), 'bar', 10)
    fits_headers |= make_fits_headers('bar', 10)

    groups = spatial_association.find_components(plate_solutions, fits_headers)

    expected_groups = ['IC1318_RA_300.0_DEC_39.5', 'IC1318_RA_300.0_DEC_40.0']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(
        all(f.startswith('foo') for f in groups['IC1318_RA_300.0_DEC_39.5']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['IC1318_RA_300.0_DEC_40.0']))

  def test_two_noisy_groups(self):
    plate_solutions = make_plate_solutions((307, 40), 'foo', 10, dither=True)
    fits_headers = make_fits_headers('foo', 10)
    plate_solutions |= make_plate_solutions((315, 45), 'bar', 10, dither=True)
    fits_headers |= make_fits_headers('bar', 10)

    groups = spatial_association.find_components(plate_solutions, fits_headers)

    expected_groups = ['North America nebula', 'IC1318']
    self.assertCountEqual(groups.keys(), expected_groups)

    self.assertTrue(all(f.startswith('foo') for f in groups['IC1318']))
    self.assertTrue(
        all(f.startswith('bar') for f in groups['North America nebula']))

  def test_sadr_pano(self):
    # print('Entering test_sadr_pano')

    # centers = [
    #     (301.625, 41.37361111111111),
    #     (301.81666666666666, 38.29111111111111),
    #     (304.34583333333336, 41.44222222222222),
    #     (304.4125, 38.35472222222222),
    #     (307.06666666666666, 41.44638888888888),
    #     (307.01666666666665, 38.36083333333333),
    # ]
    # ids = ['Pane 1', 'Pane 2', 'Pane 3', 'Pane 4', 'Pane 5', 'Pane 6']

    # plate_solutions = {}
    # for id, center in zip(ids, centers):
    #   plate_solutions |= make_plate_solutions(center, id)

    # fits_headers = make_fits_headers('bar', 10)

    # groups = spatial_association.find_components(plate_solutions,
    #                                              min_component_members=1)
    # expected_groups = {
    #     'IC1311': ['Pane 1'],
    #     'IC1318_RA_301.8_DEC_38.3': ['Pane 2'],
    #     'IC1318_RA_304.3_DEC_41.4': ['Pane 3'],
    #     'IC1318_RA_304.4_DEC_38.4': ['Pane 4'],
    #     'IC1318_RA_307.1_DEC_41.4': ['Pane 5'],
    #     'IC1318_RA_307.0_DEC_38.4': ['Pane 6'],
    # }
    # self.assertCountEqual(groups, expected_groups)

  def test_stability(self):
    plate_solutions = cache['plate_solutions']
    fits_headers = cache['fits_headers']
    expected = sorted(
        spatial_association.find_components(plate_solutions, fits_headers))

    for i in range(100):
      print(f'### Testing stability, iteration {i}')
      actual = spatial_association.find_components(plate_solutions,
                                                   fits_headers)
      self.assertEqual(sorted(actual), expected)
