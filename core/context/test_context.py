import unittest
import ray
import numpy as np
from objsize import get_deep_size
import time

from context import get_cache
from context import context, WriteMode


@ray.remote
def foo():
  assert context.bar == 123


class TestContext(unittest.TestCase):

  def setUp(self):
    context.clear()
    cache = get_cache()
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

    ref_dict = context.get_ref_dict()
    self.assertEqual(len(ref_dict), 1)
    self.assertTrue(all(type(v) is ray.ObjectRef for v in ref_dict.values()))
    self.assertLess(get_deep_size(ref_dict), 1000)

    cache = get_cache()
    self.assertEqual(list(cache.iterkeys()), ['biggie'])
    self.assertEqual(type(cache['biggie']), np.ndarray)

  def test_flush_large_object(self):
    large_object = np.random.random((1000, 1000))

    context.set('biggie', large_object, write_mode=WriteMode.DISK_ONLY)

    ref_dict = context.get_ref_dict()
    self.assertEqual(len(ref_dict), 0)

    cache = get_cache()
    self.assertEqual(list(cache.iterkeys()), ['biggie'])
    self.assertEqual(type(cache['biggie']), np.ndarray)

  def test_load_from_cache(self):
    cache = get_cache()
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
    cache = get_cache()
    cache['foo'] = 'bar'
    self.assertTrue(context.contains('foo'))

  def test_remote_fn(self):
    context.bar = 123
    ray.get(foo.remote())

  def test_lazy_load_dict(self):
    context.add_lazy_load_dict('lazy')

    context.lazy = {'abc123': 'value_abc123', 'def789': 'value_def789'}

    cache = get_cache()
    self.assertEqual(cache['lazy'], ['abc123', 'def789'])
    self.assertEqual(cache[('lazy', 'abc123')], 'value_abc123')
    self.assertEqual(cache[('lazy', 'def789')], 'value_def789')

    self.assertEqual(list(context.lazy.keys()), ['abc123', 'def789'])
    self.assertEqual(list(context.lazy.values()),
                     ['value_abc123', 'value_def789'])
    self.assertEqual(context.lazy['abc123'], 'value_abc123')
    self.assertEqual(context.lazy['def789'], 'value_def789')

    context.or_equals('lazy', {'xyz999': 'value_xyz999'})

    self.assertEqual(cache['lazy'], ['abc123', 'def789', 'xyz999'])
    self.assertEqual(cache[('lazy', 'abc123')], 'value_abc123')
    self.assertEqual(cache[('lazy', 'def789')], 'value_def789')
    self.assertEqual(cache[('lazy', 'xyz999')], 'value_xyz999')

    self.assertEqual(list(context.lazy.keys()), ['abc123', 'def789', 'xyz999'])
    self.assertEqual(list(context.lazy.values()),
                     ['value_abc123', 'value_def789', 'value_xyz999'])

    context.or_equals('lazy', {
        'mno567': 'value_mno567',
        'xyz999': 'new_value_xyz999',
    })

    self.assertEqual(cache['lazy'], ['abc123', 'def789', 'mno567', 'xyz999'])
    self.assertEqual(cache[('lazy', 'abc123')], 'value_abc123')
    self.assertEqual(cache[('lazy', 'def789')], 'value_def789')
    self.assertEqual(cache[('lazy', 'mno567')], 'value_mno567')
    self.assertEqual(cache[('lazy', 'xyz999')], 'new_value_xyz999')

    self.assertEqual(list(context.lazy.keys()),
                     ['abc123', 'def789', 'mno567', 'xyz999'])
    self.assertEqual(
        list(context.lazy.values()),
        ['value_abc123', 'value_def789', 'value_mno567', 'new_value_xyz999'])
