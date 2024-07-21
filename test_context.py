import ray_local

import unittest
import ray
import numpy as np
from objsize import get_deep_size
import datetime
import time

from context import context
from cache import cache
from testing.make_test_frames import make_synthetic_frames, copy_testdata_frames
from testing.make_test_context import make_test_context
from metadata.image_types import is_light_frame, is_dark_frame


@ray.remote
def foo():
  assert context.bar == 123


class TestContext(unittest.TestCase):

  def setUp(self):
    context.clear()
    cache.clear()

  def test_set_and_get(self):
    context.foo = 'bar'
    self.assertEqual(context.foo, 'bar')

  def test_increment(self):
    context.foo = 5
    context.foo += 1
    self.assertEqual(context.foo, 6)

  def test_set_and_get_tuple(self):
    context.set(('foo', 123), 'bar')
    self.assertEqual(context.get(('foo', 123)), 'bar')

  def test_non_existent(self):
    context.foo = 'bar'
    self.assertEqual(context.baz, None)

  def test_large_object(self):
    large_object = np.random.random((1000, 1000))
    self.assertGreater(get_deep_size(large_object), 10000)

    context.biggie = large_object
    assert type(context.biggie) is np.ndarray

    ref_dict = context.ref_dict()
    self.assertEqual(len(ref_dict), 1)
    self.assertTrue(all(type(v) is ray.ObjectRef for v in ref_dict.values()))
    self.assertLess(get_deep_size(ref_dict), 1000)

    self.assertEqual(list(cache.iterkeys()), ['biggie'])
    self.assertEqual(type(cache['biggie']), np.ndarray)

  def test_flush_large_object(self):
    large_object = np.random.random((1000, 1000))

    context.set('biggie', large_object, flush=True)

    ref_dict = context.ref_dict()
    self.assertEqual(len(ref_dict), 0)

    self.assertEqual(list(cache.iterkeys()), ['biggie'])
    self.assertEqual(type(cache['biggie']), np.ndarray)

  def test_load_from_cache(self):
    cache['foo'] = 'bar'
    self.assertEqual(context.foo, 'bar')

  def test_or_equals(self):
    context.foo = {'a': 1}
    context.or_equals('foo', {'b': 2})
    self.assertEqual(context.foo, {'a': 1, 'b': 2})

  def test_and_equals(self):
    context.foo = ['a']
    context.and_equals('foo', ['b'])
    self.assertEqual(context.foo, ['a', 'b'])

  def test_default_value(self):
    value = context.get('non-existent', {'foo': 'bar'})
    self.assertEqual(value, {'foo': 'bar'})

  def test_contains(self):
    self.assertFalse(context.contains('foo'))
    cache['foo'] = 'bar'
    self.assertTrue(context.contains('foo'))

  def test_remote_fn(self):
    context.bar = 123
    ray.get(foo.remote())
