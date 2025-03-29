import ray
import time
import xxhash
import dill
import pickle
import inspect
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional, Protocol, get_origin, TypeVar, Generic, runtime_checkable

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


# --- Injection marker ---


class Injected(Generic[TypeVar('T')]):
  pass


class MissingInjectedValueError(Exception):
  pass


# --- NodeResult wrapper ---


class NodeResult:

  def __init__(self,
               value: Any = None,
               exc: Exception = None,
               tb: Optional[str] = None):
    self.ok = exc is None
    self.value = value
    self.exc = exc
    self.tb = tb

  @classmethod
  def from_exception(cls, exc: Exception):
    import traceback
    return cls(exc=exc, tb=traceback.format_exc())

  def unwrap(self):
    if self.ok:
      return self.value
    raise self.exc


# --- Utilities ---


def compute_call_hash(serialized_func: bytes, args: tuple, kwargs: dict) -> str:
  payload = pickle.dumps((serialized_func, args, kwargs))
  return xxhash.xxh64(payload).hexdigest()


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


def _try_get_cached_result(
    func_name: str, call_hash: str, cache: CacheProtocol,
    content_hash: ContentHashProtocol) -> Optional[NodeResult]:
  if call_hash in cache:
    cached_result = cache[call_hash]
    if isinstance(cached_result.value, Path):
      if _is_valid_cached_path(cached_result.value, call_hash, content_hash):
        debug_print(f"[cache hit] {func_name} -> {call_hash}")
        return cached_result
      else:
        debug_print(f"[cache invalid] {func_name} -> {call_hash}")
    else:
      debug_print(f"[cache hit] {func_name} -> {call_hash}")
      return cached_result
  return None


def _store_result(call_hash: str, result: NodeResult, cache: CacheProtocol,
                  content_hash: ContentHashProtocol):
  debug_print(
      f"[store result] hash={call_hash}, type={type(result.value).__name__}")
  cache[call_hash] = result
  if isinstance(result.value, Path) and result.value.exists():
    content_hash.set(result.value, call_hash)


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


def _execute(func: Callable, bound: inspect.BoundArguments, call_hash: str,
             cache: CacheProtocol, content_hash: ContentHashProtocol,
             fn_name: str, cache_results: bool) -> NodeResult:
  try:
    debug_print(f"[run] {fn_name}")
    start = time.perf_counter()
    result = func(*bound.args, **bound.kwargs)
    elapsed = time.perf_counter() - start
    result_wrapper = NodeResult(value=result)
    _update_performance(fn_name, elapsed, cache)
    if cache_results:
      _store_result(call_hash, result_wrapper, cache, content_hash)
    return result_wrapper
  except Exception as e:
    return NodeResult.from_exception(e)


# --- Decorator factory ---


def make_task_decorator(cache: CacheProtocol,
                        content_hash: ContentHashProtocol):

  def task(_func=None, *, task_config: Optional[dict] = None):

    def decorator(func: Callable):
      cache_results = True if task_config is None else task_config.get(
          "cache_results", True)
      serialized_func = dill.dumps(func)

      @wraps(func)
      def wrapper(*args, injected_context: Optional[dict] = None, **kwargs):
        debug_print(f"[call] local: {func.__name__}")
        bound = _bind_args(func, args, kwargs, injected_context)
        call_hash = compute_call_hash(serialized_func, bound.args, bound.kwargs)
        debug_print(f"[hash] local: {func.__name__} -> {call_hash}")

        if cache_results:
          cached_result = _try_get_cached_result(func.__name__, call_hash,
                                                 cache, content_hash)
          if cached_result is not None:
            return cached_result

        return _execute(func, bound, call_hash, cache, content_hash,
                        func.__name__, cache_results)

      wrapper._original = func
      wrapper._is_task = True
      return wrapper

    if _func is None:
      return decorator
    return decorator(_func)

  def remote_task_decorator(_func=None, **ray_opts):
    task_config = ray_opts.pop("task_config", {})
    cache_results = task_config.get("cache_results", True)

    def decorator(func: Callable):
      serialized_func = dill.dumps(func)

      @wraps(func)
      def wrapped_for_ray(*args, **kwargs):
        debug_print(f"[execute] remote: {func.__name__}")
        bound = _bind_args(func, args, kwargs, injected_context=None)
        call_hash = compute_call_hash(serialized_func, bound.args, bound.kwargs)
        debug_print(f"[hash] remote-exec: {func.__name__} -> {call_hash}")
        return _execute(func, bound, call_hash, cache, content_hash,
                        func.__name__, cache_results)

      ray_func = ray.remote(wrapped_for_ray) if not ray_opts else ray.remote(
          **ray_opts)(wrapped_for_ray)

      @wraps(func)
      def remote_wrapper(*args,
                         injected_context: Optional[dict] = None,
                         **kwargs):
        debug_print(f"[call] remote: {func.__name__}")
        bound = _bind_args(func, args, kwargs, injected_context)
        call_hash = compute_call_hash(serialized_func, bound.args, bound.kwargs)
        debug_print(f"[hash] remote-call: {func.__name__} -> {call_hash}")

        if cache_results:
          cached_result = _try_get_cached_result(func.__name__, call_hash,
                                                 cache, content_hash)
          if cached_result is not None:
            return ray.put(cached_result)

        debug_print(f"[remote execute] {func.__name__} -> {call_hash}")
        return ray_func.remote(*bound.args, **bound.kwargs)

      remote_wrapper.remote = remote_wrapper
      remote_wrapper.options = ray_func.options
      remote_wrapper._original = func
      remote_wrapper._is_task = True
      return remote_wrapper

    if callable(_func):
      return decorator(_func)
    return decorator

  task.remote = remote_task_decorator
  return task
