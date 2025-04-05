import ray
import time
import xxhash
import inspect
import pickle
from functools import wraps
from typing import Any, NewType, Callable, Optional, Protocol, get_origin, TypeVar, Generic, runtime_checkable, List, Type

from hashing import hash_list

_MAX_PERFORMANCE_SAMPLES = 100


@runtime_checkable
class CacheProtocol(Protocol):

  def __getitem__(self, key: str) -> Any:
    ...

  def __setitem__(self, key: str, value: Any) -> None:
    ...

  def get(self, key: str, default: Any = None) -> Any:
    ...

  def __contains__(self, key: str) -> bool:
    ...


ContentHash = NewType('ContentHash', str)


@runtime_checkable
class ContentHashProtocol(Protocol):

  def has(self, obj: Any) -> bool:
    ...

  def get(self, obj: Any) -> ContentHash:
    ...

  def set(self, obj: Any, content_hash: ContentHash) -> Any:
    ...


class Injected(Generic[TypeVar('T')]):
  pass


class MissingInjectedValueError(Exception):
  pass


def hash_source_code(func: Callable) -> bytes:
  return xxhash.xxh64(inspect.getsource(func)).hexdigest()


def default_call_hasher(fn_hash: bytes, *args, **kwargs) -> str:
  data = pickle.dumps((fn_hash, args, sorted(kwargs.items())))
  return xxhash.xxh64(data).hexdigest()


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


def _update_performance(fn_name: str, elapsed_time: float,
                        cache: CacheProtocol):
  node_ip = ray.util.get_node_ip_address()
  key = ('performance_history', node_ip, fn_name)
  history = cache.get(key, [])
  history = history[-(_MAX_PERFORMANCE_SAMPLES - 1):] + [elapsed_time]
  cache[key] = history


def make_task_decorator(cache: CacheProtocol,
                        content_hasher: ContentHashProtocol,
                        call_hasher: Optional[Callable[[bytes, Any],
                                                       str]] = None,
                        no_cache_types: Optional[List[Type]] = None,
                        debug=True):
  call_hasher = call_hasher or default_call_hasher
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

      def inner(*args, injected_context: Optional[dict] = None, **kwargs):
        bound = _bind_args(func, args, kwargs, injected_context)
        call_hash = call_hasher(fn_hash, *bound.args, **bound.kwargs)

        cache_key = (func.__name__, call_hash)
        if self.cache_results and cache_key in cache:
          cached_result = cache[cache_key]
          content_hash = content_hasher.get(cached_result)
          if not content_hasher.has(cached_result) or content_hash == call_hash:
            debug_print(f'cache hit: {cache_key}')
            return cached_result
          else:
            debug_print(f'cache miss (mismatch {content_hash}): {cache_key}')
        else:
          debug_print(f'cache miss: {cache_key}' if self.
                      cache_results else f'never cache: {cache_key}')

        start = time.perf_counter()
        result = func(*bound.args, **bound.kwargs)
        elapsed = time.perf_counter() - start

        _update_performance(func.__name__, elapsed, cache)

        if self.cache_results and not isinstance(
            result, no_cache_types) and not isinstance(result, Exception):
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

      return wrapper

  class Task:

    def remote(self, _func=None, **ray_opts):
      cache = ray_opts.pop('cache', True)
      decorator = TaskDecorator(cache=cache, **ray_opts)
      if callable(_func):
        return decorator(_func)
      return decorator

  return Task()
