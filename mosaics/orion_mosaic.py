import datetime

from context import context
from testing.test_frames import copy_testdata_frames, make_synthetic_frames


def add_orion_mosaic_images():
  copy_testdata_frames(target_name='Orion Pane 1',
                       start_datetime=datetime.datetime(2024, 2, 5, 16, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 2',
                       start_datetime=datetime.datetime(2024, 2, 5, 17, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 3',
                       start_datetime=datetime.datetime(2024, 2, 5, 18, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)
  copy_testdata_frames(target_name='Orion Pane 4',
                       start_datetime=datetime.datetime(2024, 2, 5, 19, 29, 43),
                       dest_dir=context.input_dirs[0],
                       add_noise=False)

  make_synthetic_frames(prefix='dark',
                        number=5,
                        image_type='Dark',
                        start_datetime=datetime.datetime(
                            2024, 2, 4, 18, 29, 43),
                        fill_value=100,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        exposure=300,
                        dest_dir=context.input_dirs[0])
  make_synthetic_frames(prefix='bias',
                        number=7,
                        image_type='Bias',
                        start_datetime=datetime.datetime(
                            2024, 2, 5, 18, 29, 43),
                        fill_value=50,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        exposure=300,
                        dest_dir=context.input_dirs[0])
  make_synthetic_frames(prefix='flat',
                        number=11,
                        image_type='Flat',
                        start_datetime=datetime.datetime(
                            2024, 2, 6, 18, 29, 43),
                        fill_value=50,
                        instrument='ZWO ASI6200MM Pro',
                        width_pixels=9576,
                        height_pixels=6388,
                        temperature=-20,
                        exposure=300,
                        dest_dir=context.input_dirs[0])
