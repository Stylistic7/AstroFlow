import datetime
import pytz
import unittest

from runs import get_runs


# TODO: Unify the various copies of this.
def _make_test_frames(start_datetime, number=10, prefix='foo'):
  fits_headers = {}
  frames = []
  for i in range(number):
    filename = f'{prefix}_{i}.fit'
    date_obs = start_datetime + i * datetime.timedelta(minutes=5)
    fits_headers[filename] = {'DATE-OBS': date_obs}
    frames.append(filename)

  return frames, fits_headers


class TestRuns(unittest.TestCase):
  def test_get_runs_one_run(self):
    frames, fits_headers = _make_test_frames(
        pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)))

    runs = get_runs(frames, fits_headers)

    expected_runs = [(pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
                      pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 45)))]
    self.assertEqual(runs, expected_runs)

  def test_get_runs_two_close_runs(self):
    frames = []
    fits_headers = {}

    # Make 50 minutes of test frames.
    f, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 0, 0)),
                             prefix='foo')
    frames.extend(f)
    fits_headers |= h

    # Make 50 minutes of test frames, starting one hour later.
    f, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 1, 0)),
                             prefix='bar')
    frames.extend(f)
    fits_headers |= h

    runs = get_runs(frames, fits_headers)

    expected_runs = [(pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
                      pytz.utc.localize(datetime.datetime(2023, 7, 3, 1, 45)))]
    self.assertEqual(runs, expected_runs)

  def test_get_runs_two_distant_runs(self):
    frames = []
    fits_headers = {}

    # Make 50 minutes of test frames.
    f, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 0, 0)),
                             prefix='foo')
    frames.extend(f)
    fits_headers |= h

    # Make 50 minutes of test frames, starting three hours later.
    f, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 3, 0)),
                             prefix='bar')
    frames.extend(f)
    fits_headers |= h

    runs = get_runs(frames, fits_headers)

    expected_runs = [(pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
                      pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 45))),
                     (pytz.utc.localize(datetime.datetime(2023, 7, 3, 3, 0)),
                      pytz.utc.localize(datetime.datetime(2023, 7, 3, 3, 45)))]
    self.assertEqual(runs, expected_runs)