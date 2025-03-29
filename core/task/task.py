import ray
import time
import xxhash
import inspect
import pickle
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional, Protocol, get_origin, TypeVar, Generic, runtime_checkable, List, Type

# --- Constants ---
_MAX_PERFORMANCE_SAMPLES = 100
DEBUG_TASKS = False


def debug_print(*args, **kwargs):
  if DEBUG_TASKS:
    print(*args, **kwargs)


# --- Protocols ---


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


@runtime_checkable
class ContentHashProtocol(Protocol):

  def get(self, path: Path) -> str:
    ...

  def set(self, path: Path, content_hash: str) -> None:
    ...


class Injected(Generic[TypeVar('T')]):
  pass


class MissingInjectedValueError(Exception):
  pass


# --- Utilities ---


def serialize_function(func: Callable) -> bytes:
  code = func.__code__
  m = xxhash.xxh64()
  attrs = [code.co_code, code.co_consts, code.co_varnames]
  for attr in attrs:
    m.update(str(attr).encode())
  return m.digest()


def default_call_hasher(serialized_func: bytes, *args, **kwargs) -> str:
  data = pickle.dumps((serialized_func, args, sorted(kwargs.items())))
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


def _is_valid_cached_path(path: Path, call_hash: str,
                          content_hash: ContentHashProtocol) -> bool:
  try:
    return path.exists() and content_hash.get(path) == call_hash
  except Exception as e:
    debug_print(f"[cache check error] {path}: {e}")
    return False


def _update_performance(fn_name: str, elapsed_time: float,
                        cache: CacheProtocol):
  try:
    node_ip = ray.util.get_node_ip_address()
    key = ('performance_history', node_ip, fn_name)
    history = cache.get(key, [])
    history = history[-(_MAX_PERFORMANCE_SAMPLES - 1):] + [elapsed_time]
    cache[key] = history
    debug_print(f"[perf] {fn_name} ran in {elapsed_time:.3f}s on {node_ip}")
  except Exception as e:
    debug_print(f"[perf] failed for {fn_name}: {e}")


# --- Decorator factory ---


def make_task_decorator(cache: CacheProtocol,
                        content_hash: ContentHashProtocol,
                        call_hasher: Optional[Callable[[bytes, Any],
                                                       str]] = None,
                        no_cache_types: Optional[List[Type]] = None):
  call_hasher = call_hasher or default_call_hasher
  no_cache_types = tuple(no_cache_types or [])

  class TaskDecorator:

    def __init__(self, cache=True, **ray_opts):
      self.cache_results = cache
      self.ray_opts = ray_opts

    def __call__(self, func: Callable):
      serialized_func = serialize_function(func)

      def inner(*args, injected_context: Optional[dict] = None, **kwargs):
        bound = _bind_args(func, args, kwargs, injected_context)
        call_hash = call_hasher(serialized_func, *bound.args, **bound.kwargs)

        cache_key = (func.__name__, call_hash)
        if self.cache_results and cache_key in cache:
          cached_result = cache[cache_key]
          if isinstance(cached_result, Path):
            if _is_valid_cached_path(cached_result, call_hash, content_hash):
              debug_print(f"[cache hit remote] {func.__name__}")
              return cached_result
          else:
            debug_print(f"[cache hit remote] {func.__name__}")
            return cached_result

        start = time.perf_counter()
        result = func(*bound.args, **bound.kwargs)
        elapsed = time.perf_counter() - start

        _update_performance(func.__name__, elapsed, cache)

        if self.cache_results and not isinstance(result, no_cache_types):
          cache[cache_key] = result
          if isinstance(result, Path) and result.exists():
            content_hash.set(result, call_hash)

        return result

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
