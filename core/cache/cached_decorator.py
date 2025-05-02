import ray
import inspect
from functools import wraps
from typing import Any, Callable, Optional, get_origin, TypeVar, Generic, List, Type
import pickle
import xxhash

from dependency_graph import Graph
from hashing import hash_source_code, hash_call
from protocols import CacheProtocol, ContentHashProtocol


class Injected(Generic[TypeVar('T')]):
  pass


class MissingInjectedValueError(Exception):
  pass


def _bind_args(func: Callable, args: tuple, kwargs: dict,
               injected_context: Optional[dict]) -> inspect.BoundArguments:
  sig = inspect.signature(func)
  bound = sig.bind_partial(*args, **kwargs)
  bound.apply_defaults()
  for name, param in sig.parameters.items():
    if get_origin(param.annotation) == Injected and name not in bound.arguments:
      if injected_context is None or name not in injected_context:
        raise MissingInjectedValueError(f"Missing injected argument: {name}")
      bound.arguments[name] = injected_context[name]
  return bound


# TODO: Consider removing `injected_context` if it's not used anywhere. The
# original intent was to handle context for subgraphs, like temporary
# directories.
def make_cached_decorator(cache: CacheProtocol,
                          content_hasher: ContentHashProtocol,
                          call_hasher: Optional[Callable[[bytes, Any], str]],
                          debug=True):

  def debug_print(*args, **kwargs):
    if debug:
      print(*args, **kwargs)

  def decorate(func: Callable, ray_opts: dict) -> Callable:
    fn_hash = hash_source_code(func)

    def _get_cache_key_and_result(bound):
      call_hash = call_hasher(fn_hash, *bound.args, **bound.kwargs)
      key = (func.__name__, call_hash)

      if key not in cache:
        debug_print(f'cache miss: {key}')
        return key, call_hash, False, None

      cached = cache[key]
      stored_hash = content_hasher.get(cached)

      if not content_hasher.has(cached) or stored_hash == call_hash:
        debug_print(f'cache hit: {key}')
        return key, call_hash, True, cached
      else:
        debug_print(f'cache mismatch ({stored_hash}): {key}')
        return key, call_hash, False, None

    # TODO: @wraps(func) would be nice here, but it messes up the signature
    # inspection used to inject context.
    def inner(*args, injected_context: Optional[dict] = None, **kwargs):
      bound = _bind_args(func, args, kwargs, injected_context)
      key, call_hash, valid, cached_result = _get_cache_key_and_result(bound)

      if valid:
        return cached_result

      result = func(*bound.args, **bound.kwargs)

      # TODO: Remove this assert.
      assert not isinstance(result, Graph)

      if not isinstance(result, Exception):
        result = content_hasher.set(result, call_hash)
        debug_print(f'storing: {key} as {type(result)}')
        cache[key] = result

      return result

    ray_func = ray.remote(**ray_opts)(inner) if ray_opts else ray.remote(inner)

    @wraps(func)
    def wrapper(*args, **kwargs):
      raise RuntimeError(
          f"{func.__name__} is a remote task. Use `.remote(...)`.")

    wrapper.remote = ray_func.remote
    wrapper.options = ray_func.options
    wrapper._original = func
    wrapper._function = func
    wrapper._is_task = True

    for attr in ('filename_fn', 'thumbnail_fn'):
      if hasattr(func, attr):
        setattr(wrapper, attr, getattr(func, attr))

    def is_cached(*args,
                  injected_context: Optional[dict] = None,
                  **kwargs) -> bool:
      bound = _bind_args(func, args, kwargs, injected_context)
      _, _, valid, _ = _get_cache_key_and_result(bound)
      return valid

    wrapper.is_cached = is_cached

    return wrapper

  class CachedTask:

    def __call__(self, func: Callable) -> Callable:
      return decorate(func, ray_opts={})

    def remote(self, func: Callable = None, **ray_opts):
      if func is None:
        return lambda f: decorate(f, ray_opts)
      return decorate(func, ray_opts)

  return CachedTask()
