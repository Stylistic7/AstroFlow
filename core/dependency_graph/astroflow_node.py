import ray
import inspect
from pathlib import Path
from functools import wraps
import structlog
from typing import Any
import astropy.io.fits as pyfits  # type: ignore

from context import context
from artifacts import FileArtifact, artifact_exists
from hashing import hash_list
from dependency_graph.logging import capture_logs
from profiling import track_performance

log = structlog.get_logger()


def AstroFlowNode(resources=None, memory=None):

  if resources == None:
    resources = {}

  def decorator(fn):
    source = inspect.getsource(fn)

    def _result_exists(content_hash):
      key = (fn.__name__, content_hash)
      if not context.contains(key):
        return False

      result = context[key]
      if isinstance(result, Path):
        return artifact_exists(FileArtifact(result, content_hash))
      else:
        return True

    def _store_result(result, content_hash):
      key = (fn.__name__, content_hash)
      context[key] = result
      print(f'### stored result with key {key}')

      if isinstance(result, Path):
        with pyfits.open(str(result), mode='update', mmap=False) as fits:
          fits[0].header['CONTHASH'] = content_hash
        print(f'### Updated filename {result} with content hash {content_hash}')

    @ray.remote(scheduling_strategy="SPREAD",
                resources=resources,
                memory=memory)
    @capture_logs
    @track_performance
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
      log.info("function_entry",
               function_name=fn.__name__,
               args=args,
               kwargs=kwargs)

      kwargs_list = list(kwargs.items())
      content_hash = hash_list([source, *args, *kwargs_list])
      print(f'### content_hash {content_hash}')

      if _result_exists(content_hash):
        # TODO: Eliminate the double lookup.
        print(f'### Not running fn {fn.__name__} because result exists')
        key = (fn.__name__, content_hash)
        return context[key]
      else:
        print(f'### Running fn {fn.__name__}')
        result = fn(*args, **kwargs)
        _store_result(result, content_hash)
        return result

    return wrapper

  return decorator
