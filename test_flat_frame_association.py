import unittest
import tempfile
import datetime

import file_io
import make_test_frames
import flat_frame_association


class TestFlatFrameAssociation(unittest.TestCase):
  def test_same_day(self):
    light_frame_dates = [datetime.datetime(2023, 7, 3, 0, 0)]
    flat_frame_dates = [datetime.datetime(2023, 7, 3, 0, 0)]

    associations = flat_frame_association.associate_dates(
        light_frame_dates, flat_frame_dates)

    expected_associations = {datetime.datetime(2023, 7, 3, 0, 0): 0}
    self.assertEqual(associations, expected_associations)

  def test_prev_day_flats(self):
    light_frame_dates = [
        datetime.datetime(2023, 7, 4, 0, 0),
        datetime.datetime(2023, 7, 5, 0, 0)
    ]
    flat_frame_dates = [datetime.datetime(2023, 7, 3, 0, 0)]

    associations = flat_frame_association.associate_dates(
        light_frame_dates, flat_frame_dates)

    expected_associations = {
        datetime.datetime(2023, 7, 4, 0, 0): 0,
        datetime.datetime(2023, 7, 5, 0, 0): 0
    }
    self.assertEqual(associations, expected_associations)

  def test_two_flat_days(self):
    light_frame_datese = [
        datetime.datetime(2023, 7, 4, 0, 0),
        datetime.datetime(2023, 7, 5, 0, 0),
        datetime.datetime(2023, 7, 6, 0, 0),
        datetime.datetime(2023, 7, 7, 0, 0)
    ]
    flat_frame_dates = [
        datetime.datetime(2023, 7, 3, 0, 0),
        datetime.datetime(2023, 7, 6, 0, 0)
    ]

    associations = flat_frame_association.associate_dates(
        light_frame_datese, flat_frame_dates)

    expected_associations = {
        datetime.datetime(2023, 7, 4, 0, 0): 0,
        datetime.datetime(2023, 7, 5, 0, 0): 0,
        datetime.datetime(2023, 7, 6, 0, 0): 1,
        datetime.datetime(2023, 7, 7, 0, 0): 1
    }
    self.assertEqual(associations, expected_associations)


if __name__ == '__main__':
  unittest.main()
