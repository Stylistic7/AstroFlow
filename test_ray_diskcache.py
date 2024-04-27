import ray_local

import unittest
import ray

import ray_diskcache
from cache import cache


@ray.remote
def my_remote_fn():
  ray_diskcache.set('hello', 'aloha')


def my_local_fn():
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

  def test_or_equals_existing(self):
    ray_diskcache.set('foo', {'a': 'b'})
    ray_diskcache.or_equals('foo', {'a': 'd'})
    self.assertEqual(ray_diskcache.get('foo'), {'a': 'd'})

  def test_and_equals(self):
    ray_diskcache.set('foo', 0)
    ray_diskcache.and_equals('foo', 1)
    ray_diskcache.and_equals('foo', 1)
    ray_diskcache.and_equals('foo', 1)
    ray_diskcache.and_equals('foo', 1)
    self.assertEqual(ray_diskcache.get('foo'), 4)


if __name__ == "__main__":
  unittest.main()
