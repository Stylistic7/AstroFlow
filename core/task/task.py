import dill
import pickle
import traceback
import inspect
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional, Union, TypeVar, Generic, Protocol, runtime_checkable, get_origin
import xxhash


@runtime_checkable
class CacheProtocol(Protocol):

  def __getitem__(self, key: str) -> Any:
    ...

  def __setitem__(self, key: str, value: Any) -> None:
    ...


@runtime_checkable
class ContentHashProtocol(Protocol):

  def get(self, path: Path) -> str:
    ...

  def set(self, path: Path, content_hash: str) -> None:
    ...


class Injected(Generic[TypeVar('T')]):
  """Marker to indicate a task argument should be injected from graph context."""
  pass


class MissingInjectedValueError(Exception):
  pass


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
    return cls(exc=exc, tb=traceback.format_exc())

  def unwrap(self):
    if self.ok:
      return self.value
    raise self.exc


def make_task_decorator(cache: CacheProtocol,
                        content_hash: ContentHashProtocol):

  def task(_func=None, *, cache_results: bool = True):

    def decorator(func: Callable):

      @wraps(func)
      def wrapper(*args, context: Optional[dict] = None, **kwargs):
        try:
          bound = _bind_args(func, args, kwargs, context)
          call_hash = hash_func(func, bound.args, bound.kwargs)

          if not cache_results:
            return _run_task(func, bound.args, bound.kwargs)

          if call_hash in cache:
            return NodeResult(value=cache[call_hash])

          result = _run_task(func, bound.args, bound.kwargs)

          if result.ok:
            _store_result(call_hash, result.value, cache, content_hash)

          return result
        except Exception as e:
          return NodeResult.from_exception(e)

      wrapper._original = func
      wrapper._is_task = True
      return wrapper

    if _func is None:
      return decorator
    return decorator(_func)

  return task


def _bind_args(func: Callable, args: tuple, kwargs: dict,
               context: Optional[dict]) -> inspect.BoundArguments:
  sig = inspect.signature(func)
  bound = sig.bind_partial(*args, **kwargs)
  bound.apply_defaults()

  for name, param in sig.parameters.items():
    if get_origin(param.annotation) == Injected and name not in bound.arguments:
      if context is None or name not in context:
        raise MissingInjectedValueError(f"Missing injected argument: {name}")
      bound.arguments[name] = context[name]

  return bound


def _run_task(func: Callable, args: tuple, kwargs: dict) -> NodeResult:
  try:
    return NodeResult(value=func(*args, **kwargs))
  except Exception as e:
    return NodeResult.from_exception(e)


def _store_result(call_hash: str, result: Any, cache: CacheProtocol,
                  content_hash: ContentHashProtocol):
  cache[call_hash] = result

  if isinstance(result, Path):
    assert result.exists()
    content_hash.set(result, call_hash)


def hash_func(func: Callable, args: tuple, kwargs: dict) -> str:
  serialized_func = dill.dumps(func)
  payload = pickle.dumps((serialized_func, args, kwargs))
  return xxhash.xxh64(payload).hexdigest()


def validate_graph(graph: dict, context: dict):
  for node in graph:
    task_fn = node if callable(node) else None
    if not hasattr(task_fn, '_is_task'):
      continue
    sig = inspect.signature(task_fn._original)
    for name, param in sig.parameters.items():
      if get_origin(param.annotation) == Injected:
        if name not in context:
          raise MissingInjectedValueError(f"Missing injected argument: {name}")
