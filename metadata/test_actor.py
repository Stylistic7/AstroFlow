from ray_infra import ray_local

import ray
import unittest
import time
import datetime

from context import context
from compute_graph import build_compute_graph
import clustering
from dependency_graph import run_graph
from metadata.image_metadata import scatter_compute_image_metadata
from metadata.image_types import is_light_frame, is_dark_frame
from testing.test_context import make_test_context
from orion_mosaic import add_orion_mosaic_images
from metadata.actor import get_metadata, get_metadata_for_mosaic, group_by_header, sort_by_header, cluster_by_header
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

  def test_orion_mosaic(self):
    add_orion_mosaic_images()

    graph = ray.get(scatter_compute_image_metadata.remote('input'))
    run_graph(graph)

    result = get_metadata()
    self.assertEqual(len(result), 35)

    result = group_by_header('IMAGETYP')
    print('### result', result)
    print('### groups', result.keys())
    self.assertEqual(len(result), 4)
    light_frames = result['light']
    print('### light_frames', light_frames)
    self.assertEqual(len(light_frames), 12)

    sorted_result = sort_by_header(light_frames, 'DATE-OBS')
    print('### sorted_result', sorted_result)

    clustered_result = cluster_by_header(
        light_frames, 'DATE-OBS', lambda images: clustering.cluster_continuous(
            images, epsilon=datetime.timedelta(minutes=15)))
    print('### clustered_result', clustered_result)
    self.assertEqual(len(clustered_result), 4)

    clustered_result = cluster_by_header(
        light_frames, 'DATE-OBS', lambda images: clustering.cluster_continuous(
            images, epsilon=datetime.timedelta(days=2)))
    print('### clustered_result', clustered_result)
    self.assertEqual(len(clustered_result), 1)

  def test_get_metadata_for_mosaic(self):
    add_orion_mosaic_images()

    graph = build_compute_graph()
    context.regroup = True
    run_graph(graph)

    result = get_metadata()
    self.assertEqual(len(result), 35)

    mosaics = context.mosaics
    self.assertEqual(len(mosaics), 1)
    mosaic_id = next(iter(mosaics.keys()))

    result = get_metadata_for_mosaic(mosaic_id)
    self.assertEqual(len(result), 35)


if __name__ == '__main__':
  unittest.main()
