import unittest
import tempfile
import datetime
import pytz
import copy

import file_io
from make_test_frames import make_synthetic_frames
import step_flat_frame_association
from stack import Stack
import runs


class TestFlatFrameAssociation(unittest.TestCase):

  def test_same_day(self):
    _, light_f, light_p = make_synthetic_frames(
        start_datetime=pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
        prefix='light_foo')
    _, flat_f, flat_p = make_synthetic_frames(start_datetime=pytz.utc.localize(
        datetime.datetime(2023, 7, 3, 12, 0)),
                                              image_type='Flat',
                                              prefix='flat_foo')

    fits_headers = light_f | flat_f
    plate_solutions = light_p | flat_p

    light_frames = light_f.keys()
    flat_frames = flat_f.keys()

    stack = Stack(target='foo',
                  light_frames=light_frames,
                  width_pixels=3000,
                  height_pixels=2000,
                  image_scale=1.0)

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flat_frames, fits_headers)

    light_runs = runs.get_runs(light_frames, fits_headers)
    self.assertEqual(len(light_runs), 1)

    flat_runs = runs.get_runs(flat_frames, fits_headers)
    self.assertEqual(len(flat_runs), 1)

    expected = [(light_runs[0], flat_runs[0])]
    self.assertEqual(stack.flat_frame_associations, expected)

  def test_prev_day(self):
    _, light_f, light_p = make_synthetic_frames(
        start_datetime=pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
        prefix='light_foo')
    _, flat_f, flat_p = make_synthetic_frames(start_datetime=pytz.utc.localize(
        datetime.datetime(2023, 7, 2, 0, 0)),
                                              image_type='Flat',
                                              prefix='flat_foo')

    fits_headers = light_f | flat_f
    plate_solutions = light_p | flat_p

    light_frames = light_f.keys()
    flat_frames = flat_f.keys()

    stack = Stack(target='foo',
                  light_frames=light_frames,
                  width_pixels=3000,
                  height_pixels=2000,
                  image_scale=1.0)

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flat_frames, fits_headers)

    light_runs = runs.get_runs(light_frames, fits_headers)
    self.assertEqual(len(light_runs), 1)

    flat_runs = runs.get_runs(flat_frames, fits_headers)
    self.assertEqual(len(flat_runs), 1)

    expected = [(light_runs[0], flat_runs[0])]
    self.assertEqual(stack.flat_frame_associations, expected)

  def test_two_flat_days(self):
    _, light_foo_f, light_foo_p = make_synthetic_frames(
        start_datetime=pytz.utc.localize(datetime.datetime(2023, 7, 3, 0, 0)),
        prefix='light_foo')
    _, flat_foo_f, flat_foo_p = make_synthetic_frames(
        start_datetime=pytz.utc.localize(datetime.datetime(2023, 7, 2, 0, 0)),
        image_type='Flat',
        prefix='flat_foo')
    _, light_bar_f, light_bar_p = make_synthetic_frames(
        start_datetime=pytz.utc.localize(datetime.datetime(2023, 7, 5, 0, 0)),
        prefix='light_bar')
    _, flat_bar_f, flat_bar_p = make_synthetic_frames(
        start_datetime=pytz.utc.localize(datetime.datetime(2023, 7, 6, 0, 0)),
        image_type='Flat',
        prefix='flat_bar')

    fits_headers = light_foo_f | flat_foo_f | light_bar_f | flat_bar_f
    plate_solutions = light_foo_p | flat_foo_p | light_bar_p | flat_bar_p

    light_frames = list(light_foo_f.keys()) + list(light_bar_f.keys())
    flat_frames = list(flat_foo_f.keys()) + list(flat_bar_f.keys())

    stack = Stack(target='stack',
                  light_frames=light_frames,
                  width_pixels=3000,
                  height_pixels=2000,
                  image_scale=1.0)

    step_flat_frame_association.associate_flat_frames_for_stack(
        stack, flat_frames, fits_headers)

    light_runs = runs.get_runs(light_frames, fits_headers)
    self.assertEqual(len(light_runs), 2)

    flat_runs = runs.get_runs(flat_frames, fits_headers)
    self.assertEqual(len(flat_runs), 2)

    expected = [(light_runs[0], flat_runs[0]), (light_runs[1], flat_runs[1])]
    self.assertEqual(stack.flat_frame_associations, expected)


if __name__ == '__main__':
  unittest.main()
