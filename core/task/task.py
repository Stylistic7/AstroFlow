import ray
import time
import xxhash
import inspect
import pickle
from functools import wraps
from typing import Any, Callable, Optional, get_origin, TypeVar, Generic, List, Type

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


def make_task_decorator(cache: CacheProtocol,
                        content_hasher: ContentHashProtocol,
                        call_hasher: Optional[Callable[[bytes, Any],
                                                       str]] = None,
                        no_cache_types: Optional[List[Type]] = None,
                        debug=True):
  call_hasher = call_hasher or hash_call
  no_cache_types = tuple(no_cache_types or [])

  def debug_print(*args, **kwargs):
    if debug:
      print(*args, **kwargs)

  class TaskDecorator:

    def __init__(self, cache=True, **ray_opts):
      self.cache_results = cache
      self.ray_opts = ray_opts

    def __call__(self, func: Callable):
      fn_hash = hash_source_code(func)

      def _get_cache_key_and_result(bound_args):
        call_hash = call_hasher(fn_hash, *bound_args.args, **bound_args.kwargs)
        cache_key = (func.__name__, call_hash)

        if not self.cache_results:
          debug_print(f'never cache: {cache_key}')
          return cache_key, call_hash, False, None

        if cache_key not in cache:
          debug_print(f'cache miss: {cache_key}')
          return cache_key, call_hash, False, None

        cached_result = cache[cache_key]
        content_hash = content_hasher.get(cached_result)

        if not content_hasher.has(cached_result) or content_hash == call_hash:
          debug_print(f'cache hit: {cache_key}')
          return cache_key, call_hash, True, cached_result
        else:
          debug_print(f'cache miss (mismatch {content_hash}): {cache_key}')
          return cache_key, call_hash, False, None

      def inner(*args, injected_context: Optional[dict] = None, **kwargs):
        bound = _bind_args(func, args, kwargs, injected_context)
        cache_key, call_hash, is_valid_cache, cached_result = _get_cache_key_and_result(
            bound)

        if is_valid_cache:
          return cached_result

        result = func(*bound.args, **bound.kwargs)

        if (self.cache_results and not isinstance(result, no_cache_types) and
            not isinstance(result, Exception)):
          result = content_hasher.set(result, call_hash)
          debug_print(f'store result: {cache_key} result type {type(result)}')
          cache[cache_key] = result

        return result

      inner.__name__ = func.__name__
      inner.__qualname__ = func.__qualname__

      ray_remote_callable = ray.remote(
          **self.ray_opts)(inner) if self.ray_opts else ray.remote(inner)

      @wraps(func)
      def wrapper(*args, **kwargs):
        raise RuntimeError(
            f"{func.__name__} is a remote task. Call it with `.remote(...)`.")

      wrapper.remote = ray_remote_callable.remote
      wrapper.options = ray_remote_callable.options
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
        _, _, is_valid_cache, _ = _get_cache_key_and_result(bound)
        return is_valid_cache

      wrapper.is_cached = is_cached

      return wrapper

  class Task:

    def remote(self, _func=None, **ray_opts):
      cache = ray_opts.pop('cache', True)
      decorator = TaskDecorator(cache=cache, **ray_opts)
      if callable(_func):
        return decorator(_func)
      return decorator

  return Task()
