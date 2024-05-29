import ray_local

import unittest
import ray
import numpy as np
from objsize import get_deep_size

from new_context import new_context
from cache import cache


class TestNewContext(unittest.TestCase):

  def setUp(self):
    new_context.clear()
    cache.clear()

  def test_set_and_get(self):
    new_context.foo = 'bar'
    self.assertEqual(new_context.foo, 'bar')

  def test_increment(self):
    new_context.foo = 5
    new_context.foo += 1
    self.assertEqual(new_context.foo, 6)

  def test_set_and_get_tuple(self):
    new_context.set(('foo', 123), 'bar')
    self.assertEqual(new_context.get(('foo', 123)), 'bar')

  def test_non_existent(self):
    new_context.foo = 'bar'
    self.assertEqual(new_context.baz, None)

  def test_large_object(self):
    large_object = np.random.random((1000, 1000))
    self.assertGreater(get_deep_size(large_object), 10000)

    new_context.biggie = large_object
    assert type(new_context.biggie) is np.ndarray

    ref_dict = new_context.ref_dict()
    self.assertEqual(len(ref_dict), 1)
    self.assertTrue(all(type(v) is ray.ObjectRef for v in ref_dict.values()))
    self.assertLess(get_deep_size(ref_dict), 1000)

    self.assertEqual(list(cache.iterkeys()), ['biggie'])
    self.assertEqual(type(cache['biggie']), np.ndarray)

  def test_flush_large_object(self):
    large_object = np.random.random((1000, 1000))

    new_context.set('biggie', large_object)
    new_context.flush('biggie')

    ref_dict = new_context.ref_dict()
    self.assertEqual(len(ref_dict), 0)

    self.assertEqual(list(cache.iterkeys()), ['biggie'])
    self.assertEqual(type(cache['biggie']), np.ndarray)

  def test_load_from_cache(self):
    cache['foo'] = 'bar'
    self.assertEqual(new_context.foo, 'bar')

  def test_or_equals(self):
    new_context.foo = {'a': 1}
    new_context.foo |= {'b': 2}
    self.assertEqual(new_context.foo, {'a': 1, 'b': 2})
