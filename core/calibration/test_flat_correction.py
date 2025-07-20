import datetime

from context import context
from testing import make_synthetic_frames, copy_testdata_frames
from dependency_graph import run_graph
from compute_graph import build_compute_graph


def test_one_perfect_stack(test_context) -> None:
  copy_testdata_frames(prefix='light_run',
                       target_name='NGC7635',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=context.input_dirs[0])
  make_synthetic_frames(prefix='bias_run_0',
                        number=5,
                        image_type='Bias',
                        start_datetime=datetime.datetime(
                            2024, 2, 5, 18, 29, 43),
                        fill_value=50,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        dest_dir=context.input_dirs[0])
  make_synthetic_frames(prefix='dark_run_0',
                        number=7,
                        image_type='Dark',
                        start_datetime=datetime.datetime(
                            2024, 2, 5, 18, 29, 43),
                        fill_value=50,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        dest_dir=context.input_dirs[0])
  make_synthetic_frames(prefix='flat_run_0',
                        number=11,
                        image_type='Flat',
                        start_datetime=datetime.datetime(
                            2024, 2, 5, 18, 29, 43),
                        fill_value=50,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        dest_dir=context.input_dirs[0])
  make_synthetic_frames(prefix='flat_run_1',
                        number=11,
                        image_type='Flat',
                        start_datetime=datetime.datetime(
                            2024, 2, 4, 18, 29, 43),
                        fill_value=50,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        dest_dir=context.input_dirs[0])

  graph = build_compute_graph()
  run_graph(graph)

  stacks = context.stacks
  assert len(stacks) == 1
  assert len(context.calibration_stacks) == 4

  thumbnail_keys = context.get_matching(('thumbnail',))
  assert len(thumbnail_keys) == 7
