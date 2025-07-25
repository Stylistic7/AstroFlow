import pytest
import ray
import numpy as np
from objsize import get_deep_size

from context import get_cache
from context import context, WriteMode


@ray.remote
def foo():
  assert context.bar == 123


@pytest.fixture()
def empty_context(init_ray):
  assert ray.is_initialized()
  context.clear()
  cache = get_cache()
  cache.clear()


def test_set_and_get(empty_context):
  context.foo = 'bar'
  assert context.foo == 'bar'


def test_increment(empty_context):
  context.foo = 5
  context.foo += 1
  assert context.foo == 6


def test_set_and_get_tuple(empty_context):
  context.set(('foo', 123), 'bar')
  assert context.get(('foo', 123)) == 'bar'


def test_non_existent(empty_context):
  context.foo = 'bar'
  assert context.baz is None


def test_large_object(empty_context):
  large_object = np.random.random((1000, 1000))
  assert get_deep_size(large_object) > 10000

  context.biggie = large_object
  assert type(context.biggie) is np.ndarray

  ref_dict = context.get_ref_dict()
  print(f'### ref_dict keys {list(ref_dict.keys())}')
  assert len(ref_dict) == 1
  assert all(type(v) is ray.ObjectRef for v in ref_dict.values())
  assert get_deep_size(ref_dict) < 1000

  cache = get_cache()
  assert list(cache.iterkeys()) == ['biggie']
  assert type(cache['biggie']) == np.ndarray


def test_flush_large_object(empty_context):
  large_object = np.random.random((1000, 1000))

  context.set('biggie', large_object, write_mode=WriteMode.DISK_ONLY)

  ref_dict = context.get_ref_dict()
  assert len(ref_dict) == 0

  cache = get_cache()
  assert list(cache.iterkeys()) == ['biggie']
  assert type(cache['biggie']) == np.ndarray


def test_load_from_cache(empty_context):
  cache = get_cache()
  cache['foo'] = 'bar'
  assert context.foo == 'bar'


def test_or_equals(empty_context):
  context.foo = {'a': 1}
  context.or_equals('foo', {'b': 2})
  assert context.foo == {'a': 1, 'b': 2}


def test_and_equals(empty_context):
  context.foo = ['a']
  context.and_equals('foo', ['b'])
  assert context.foo == ['a', 'b']


def test_default_value(empty_context):
  value = context.get('non-existent', {'foo': 'bar'})
  assert value == {'foo': 'bar'}


def test_contains(empty_context):
  assert not context.contains('foo')
  cache = get_cache()
  cache['foo'] = 'bar'
  assert context.contains('foo')


def test_remote_fn(empty_context):
  context.bar = 123
  ray.get(foo.remote())


def test_lazy_load_dict(empty_context):
  context.add_lazy_load_dict('lazy')

  context.lazy = {'abc123': 'value_abc123', 'def789': 'value_def789'}

  cache = get_cache()
  assert cache['lazy'] == ['abc123', 'def789']
  assert cache[('lazy', 'abc123')] == 'value_abc123'
  assert cache[('lazy', 'def789')] == 'value_def789'

  assert list(context.lazy.keys()) == ['abc123', 'def789']
  assert list(context.lazy.values()) == ['value_abc123', 'value_def789']
  assert context.lazy['abc123'] == 'value_abc123'
  assert context.lazy['def789'] == 'value_def789'

  context.or_equals('lazy', {'xyz999': 'value_xyz999'})

  assert cache['lazy'] == ['abc123', 'def789', 'xyz999']
  assert cache[('lazy', 'abc123')] == 'value_abc123'
  assert cache[('lazy', 'def789')] == 'value_def789'
  assert cache[('lazy', 'xyz999')] == 'value_xyz999'

  assert list(context.lazy.keys()) == ['abc123', 'def789', 'xyz999']
  assert list(context.lazy.values()) == [
      'value_abc123', 'value_def789', 'value_xyz999'
  ]

  context.or_equals('lazy', {
      'mno567': 'value_mno567',
      'xyz999': 'new_value_xyz999',
  })

  assert cache['lazy'] == ['abc123', 'def789', 'mno567', 'xyz999']
  assert cache[('lazy', 'abc123')] == 'value_abc123'
  assert cache[('lazy', 'def789')] == 'value_def789'
  assert cache[('lazy', 'mno567')] == 'value_mno567'
  assert cache[('lazy', 'xyz999')] == 'new_value_xyz999'

  assert list(context.lazy.keys()) == ['abc123', 'def789', 'mno567', 'xyz999']
  assert list(context.lazy.values()) == [
      'value_abc123', 'value_def789', 'value_mno567', 'new_value_xyz999'
  ]
