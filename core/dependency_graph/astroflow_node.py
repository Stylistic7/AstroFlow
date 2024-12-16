import ray
import inspect
from pathlib import Path
from functools import wraps
import structlog
from typing import Any
import astropy.io.fits as pyfits  # type: ignore
from dataclasses import dataclass

from context import context, WriteMode
from artifacts import FileArtifact, artifact_exists
from hashing import hash_list
from dependency_graph import Graph
from dependency_graph.logging import capture_logs
from profiling import track_performance
from file_io.xisf_file_io import add_xisf_metadata

log = structlog.get_logger()

# What are the cases that need to be handled?
#
# Function returns a Filename(...)
# context[(fn_name, content_hash)] = Filename(...)
# wrapper returns Filename directly
#
# Function returns a LargeObject(...)
# context[(fn_name, content_hash)] = large_object, unwrapped
# wrapper returns key (fn_name, content_hash)
#   ...this is the problem: how does the wrapper know the object was a
#      a LargeObject? All it knows is that the key exists, but not what
#      kind of object it was.
#
#   I don't want to return LargeObjects to the user, because that leaks the
#   API out to every caller.
#
# Function returns a small object, naked
# context[(fn_name, content_hash)] = small_object
# wrapper returns small_object

# I can put returns_large_objects=True in the decorator params, and then the
# decorator will store anything returned by the function as a large object,
# and will always return keys instead of the large objects. This is gross
# for functions that return either Filename or large object, though.
#
# I could just have a return_key=True param, and then any downstream user just
# knows they have to retrieve the value with context[key]. Everything is
# constitent and obvious at least.

# @dataclass(frozen=True)
# class LargeObject():
#   object: Any


# TODO: Hide logs behind debug flag.
def AstroFlowNode(resources=None, memory=None, return_key=False):

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
      if isinstance(result, Graph):
        return

      key = (fn.__name__, content_hash)
      write_mode = (WriteMode.DISK_ONLY
                    if return_key else WriteMode.MEMORY_AND_DISK)
      context.set(key, result, write_mode)
      print(f'### stored result {type(result)} with key {key} with '
            f'write mode {write_mode}')

      # TODO: Clean this up.
      if isinstance(result, Path):
        if result.suffix.lower() in ['.xisf']:
          add_xisf_metadata(result, result, {'CONTHASH': content_hash})
          print(f'### Updated XISF filename {result} with '
                f'content hash {content_hash}')
        elif result.suffix.lower() in ['.fit', '.fits']:
          with pyfits.open(str(result), mode='update', mmap=False) as fits:
            fits[0].header['CONTHASH'] = content_hash
          print(f'### Updated FITS filename {result} with '
                f'content hash {content_hash}')

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
      key = (fn.__name__, content_hash)
      print(f'### key {key}')

      if _result_exists(content_hash):
        print(f'### Not running fn {fn.__name__} because result exists')
        return key if return_key else context[key]
      else:
        print(f'### Running fn {fn.__name__}')
        result = fn(*args, **kwargs)
        _store_result(result, content_hash)
        return key if return_key else result

    return wrapper

  return decorator
