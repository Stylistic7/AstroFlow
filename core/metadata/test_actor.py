from ray_infra import ray_local

import unittest
import time
import datetime

from metadata.image_types import is_light_frame, is_dark_frame
from testing.test_context import make_test_context
from metadata.actor import get_metadata
from testing.test_context import make_test_context
from testing.test_frames import make_synthetic_frames


class TestMetadataActor(unittest.TestCase):

  def setUp(self):
    self.input_dir_obj, self.output_dir_obj = make_test_context()

  def tearDown(self):
    self.input_dir_obj.cleanup()
    self.output_dir_obj.cleanup()

  def test_image_metadata(self):
    make_synthetic_frames(prefix='dark',
                          number=20,
                          image_type='Dark',
                          fill_value=100,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15)
    make_synthetic_frames(prefix='bias',
                          number=20,
                          image_type='Bias',
                          fill_value=50,
                          instrument='ZWO ASI6200MM Pro',
                          width_pixels=9576,
                          height_pixels=6388,
                          temperature=-15)

    start_datetime = datetime.datetime(2024, 2, 5, 16, 29, 43)
    for i in range(10):
      make_synthetic_frames(prefix=f'light_{i}',
                            number=10,
                            start_datetime=start_datetime,
                            image_type='Light',
                            fill_value=50,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15)
      make_synthetic_frames(prefix=f'flat_{i}',
                            number=10,
                            start_datetime=start_datetime,
                            image_type='Flat',
                            fill_value=50,
                            instrument='ZWO ASI6200MM Pro',
                            width_pixels=9576,
                            height_pixels=6388,
                            temperature=-15)

      start_datetime += datetime.timedelta(days=1)

    def light_predicate_fn(image):
      return is_light_frame(image.header)

    def dark_predicate_fn(image):
      return is_dark_frame(image.header)

    def transform_fn(image):
      return image.content_hash

    start_time = time.time()
    result = get_metadata(predicate_fn=light_predicate_fn,
                          transform_fn=transform_fn)
    end_time = time.time()
    self.assertEqual(len(result), 100)
    print('### len(result)', len(result), 'elapsed time',
          (end_time - start_time))

    start_time = time.time()
    result = get_metadata(predicate_fn=dark_predicate_fn,
                          transform_fn=transform_fn)
    end_time = time.time()
    self.assertEqual(len(result), 20)
    print('### len(result)', len(result), 'elapsed time',
          (end_time - start_time))


if __name__ == '__main__':
  unittest.main()
