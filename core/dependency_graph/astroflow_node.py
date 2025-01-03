import ray
import inspect
from pathlib import Path
from functools import wraps
import structlog
from typing import Any
import astropy.io.fits as pyfits  # type: ignore
from dataclasses import dataclass

from context import context, WriteMode
from artifacts import Artifact, FileArtifact, CacheArtifact, artifact_exists
from artifacts.artifact import unpack_artifact
from file_io.generic import get_content_hash
from hashing import hash_list
from dependency_graph import Graph
from dependency_graph.logging import capture_logs
from profiling import track_performance
from file_io.xisf_file_io import add_xisf_metadata

log = structlog.get_logger()


class ArtifactNotFoundException(Exception):
  pass


class WrongContentHashException(Exception):
  pass


# TODO: Hide logs behind debug flag.
# TODO: cache_result and always_run are maybe the same thing?
def AstroFlowNode(resources=None,
                  memory=None,
                  always_run=False,
                  update_header=True,
                  unpack_artifacts=True,
                  cache_result=True,
                  keep_in_memory=True):

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
        return result

      # TODO: Clean this up.
      if update_header and isinstance(result, Path):
        if result.suffix.lower() in ['.xisf']:
          add_xisf_metadata(result, result, {'CONTHASH': content_hash})
          print(f'### Updated XISF filename {result} with '
                f'content hash {content_hash}')
        elif result.suffix.lower() in ['.fit', '.fits']:
          with pyfits.open(str(result), mode='update', mmap=False) as fits:
            fits[0].header['CONTHASH'] = content_hash
          print(f'### Updated FITS filename {result} with '
                f'content hash {content_hash}')

        return FileArtifact(result, content_hash)

      key = (fn.__name__, content_hash)
      write_mode = (WriteMode.MEMORY_AND_DISK
                    if keep_in_memory else WriteMode.DISK_ONLY)
      context.set(key, result, write_mode)
      print(f'### stored result {type(result)} with key {key} with '
            f'write mode {write_mode}')
      return CacheArtifact(key)

    # This decorator accepts arguments that may include FileArtifacts and / or
    # CacheArtifacts. These are "unpacked" to the contained Path or arbitrary
    # object. The content hashes of Paths (files on disk) are checked. The
    # content hashes should never leak out of the decorator.
    #
    # The decorator wraps whatever the function returns in an Artifact, also.
    # If the function returns a Path, the file's header is updated with the
    # content hash, and the Path is returned inside a FileArtifact.
    #
    # For all other objects, their value is written into the context, and a
    # CacheArtifact is returned.
    #
    # When these functions are composed into compute graphs, the artifacts are
    # handled transparently. The user code never has to worry about content
    # hashes, and sees the arguments as just Paths, or whatever arbitrary
    # Python objects are returned by the functions.
    #
    # TODO: @capture_logs and @track_performance are misplaced here. They should
    # only run when the wrapped function runs. Maybe they should be folded inot
    # this single decorator.
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

      if unpack_artifacts:
        args = [unpack_artifact(a) for a in args]
        kwargs = {k: unpack_artifact(v) for k, v in kwargs.items()}

      if not cache_result:
        return fn(*args, **kwargs)

      kwargs_list = list(kwargs.items())
      content_hash = hash_list([source, *args, *kwargs_list])
      key = (fn.__name__, content_hash)
      print(f'### key {key}')

      if not always_run and _result_exists(content_hash):
        print(f'### Not running fn {fn.__name__} because result exists')
        return CacheArtifact(key)
      else:
        print(f'### Running fn {fn.__name__}')
        result = fn(*args, **kwargs)
        return _store_result(result, content_hash)

    return wrapper

  return decorator
