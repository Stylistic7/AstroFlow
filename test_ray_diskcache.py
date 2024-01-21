# yapf: disable
import os
os.environ["ASTROFLOW_MOCK_CACHE"] = "true"

from cache import cache

import ray
ray.init(
  address="local",
  namespace='AstroFlow',
  resources={"diskcache": 1},
  ignore_reinit_error=True,
  runtime_env={"env_vars": {
    "ASTROFLOW_MOCK_CACHE": "true"
  }},
)
# yapf: enable

from make_test_frames import make_test_frames
import image_metadata
import ray_diskcache
import unittest
import tempfile


@ray.remote
def my_remote_fn():
  print('### in my_remote_fn')
  ray_diskcache.set('hello', 'aloha')


def my_local_fn():
  print('### in my_local_fn')
  ray_diskcache.set('ham', 'eggs')


class TestRayDiskcache(unittest.TestCase):

  def setUp(self):
    ray_diskcache.clear()
    cache.clear()

  def tearDown(self):
    ray_diskcache.clear()
    cache.clear()

  def test_global_dict(self):
    # Check behavior when key is missing.
    foo = ray_diskcache.get("foo")
    self.assertEqual(foo, None)

    # Check calling the ray_diskcache module functions directly.
    ray_diskcache.set('foo', 'bar')
    self.assertEqual(ray_diskcache.get('foo'), 'bar')

    # Check calling them in a remote task.
    ray.get(my_remote_fn.remote())
    self.assertEqual(ray_diskcache.get('hello'), 'aloha')

  def test_or_equals(self):
    ray_diskcache.set('foo', {'a': 'b'})
    ray_diskcache.or_equals('foo', {'c': 'd'})
    self.assertEqual(ray_diskcache.get('foo'), {'a': 'b', 'c': 'd'})

  def test_plus_equals(self):
    ray_diskcache.set('foo', 0)
    ray_diskcache.plus_equals('foo', 1)
    ray_diskcache.plus_equals('foo', 1)
    ray_diskcache.plus_equals('foo', 1)
    ray_diskcache.plus_equals('foo', 1)
    self.assertEqual(ray_diskcache.get('foo'), 4)


if __name__ == "__main__":
  unittest.main()
