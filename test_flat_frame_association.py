import unittest
import tempfile
import datetime
import pytz
import copy

import file_io
import make_test_frames
import step_flat_frame_association
from stack import Stack


def _make_test_frames(start_datetime, number=10, prefix='foo'):
  fits_headers = {}
  frames = []
  for i in range(number):
    filename = f'{prefix}_{i}.fit'
    date_obs = start_datetime + i * datetime.timedelta(minutes=5)
    fits_headers[filename] = {'DATE-OBS': date_obs}
    frames.append(filename)

  return frames, fits_headers


class TestFlatFrameAssociation(unittest.TestCase):
  def test_get_runs_one_run(self):
    frames, fits_headers = _make_test_frames(
        pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)))

    runs = step_flat_frame_association.get_runs(frames, fits_headers)

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

    runs = step_flat_frame_association.get_runs(frames, fits_headers)

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

    runs = step_flat_frame_association.get_runs(frames, fits_headers)

    expected_runs = [(pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
                      pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 45))),
                     (pytz.utc.localize(datetime.datetime(2023, 7, 3, 3, 0)),
                      pytz.utc.localize(datetime.datetime(2023, 7, 3, 3, 45)))]
    self.assertEqual(runs, expected_runs)

  def test_same_day(self):
    light_frames, fits_headers = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 0, 0)),
                                                   prefix='light_foo')
    flat_frames, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 12, 0)),
                                       prefix='flat_foo')
    fits_headers |= h

    stack = Stack(target='foo', light_frames=light_frames)

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flat_frames, fits_headers)

    expected = [(light_frames, flat_frames)]

    self.assertEqual(stack.flat_frame_associations, expected)

  def test_prev_day(self):
    light_frames, fits_headers = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 0, 0)),
                                                   prefix='light_foo')
    flat_frames, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 2, 0, 0)),
                                       prefix='flat_foo')
    fits_headers |= h

    stack = Stack(target='foo', light_frames=light_frames)

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flat_frames, fits_headers)

    expected = [(light_frames, flat_frames)]

    self.assertEqual(stack.flat_frame_associations, expected)

  def test_two_flat_days(self):
    print('### Entering test_two_flat_days')
    light_foo, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 0, 0)),
                                     prefix='light_foo')
    light_frames = copy.copy(light_foo)
    fits_headers = h

    flat_foo, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 2, 0, 0)),
                                    prefix='flat_foo')
    flat_frames = copy.copy(flat_foo)
    fits_headers |= h

    light_bar, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 5, 0, 0)),
                                     prefix='light_bar')
    light_frames.extend(light_bar)
    fits_headers |= h

    flat_bar, h = _make_test_frames(pytz.utc.localize(
        datetime.datetime(2023, 7, 6, 0, 0)),
                                    prefix='flat_bar')
    flat_frames.extend(flat_bar)
    fits_headers |= h

    stack = Stack(target='stack', light_frames=light_frames)

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flat_frames, fits_headers)
    print('### associations', stack.flat_frame_associations)

    expected = [(light_foo, flat_foo), (light_bar, flat_bar)]
    print('### expected', expected)

    self.assertEqual(stack.flat_frame_associations, expected)


if __name__ == '__main__':
  unittest.main()
