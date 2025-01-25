import ray
import inspect
from pathlib import Path
from functools import wraps
import structlog
from typing import Any
import astropy.io.fits as pyfits  # type: ignore

from context import context, WriteMode
from artifacts import FileArtifact, CacheArtifact, artifact_exists
from artifacts.artifact import unpack_artifact
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
def AstroFlowNode(resources=None,
                  memory=None,
                  update_header=True,
                  unpack_artifacts=True,
                  cache_result=False,
                  keep_in_memory=True,
                  debug=False):

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
        if debug:
          print(
              f'### Checking that file {result} has content hash {content_hash}'
          )
        return artifact_exists(FileArtifact(result, content_hash))
      else:
        return True

    def _store_result(result, content_hash):
      key = (fn.__name__, content_hash)

      # Graphs are a special case. When a function returns a Graph, it
      # represents some computation that needs to be done. If the function
      # and its arguments have been seen before, it means that computation has
      # already run. Instead of serializing the Graph, we store the value None.
      # The next time the function is called, None will be returned, instead
      # of a Graph.
      if isinstance(result, Graph):
        context.set(key, None, WriteMode.MEMORY_AND_DISK)
        return result

      # TODO: Clean this up.
      if update_header and isinstance(result, Path):
        context.set(key, result)

        if result.suffix.lower() in ['.xisf']:
          add_xisf_metadata(result, result, {'CONTHASH': content_hash})
          if debug:
            print(f'### Updated XISF filename {result} with '
                  f'content hash {content_hash}')
        elif result.suffix.lower() in ['.fit', '.fits']:
          with pyfits.open(str(result), mode='update', mmap=False) as fits:
            fits[0].header['CONTHASH'] = content_hash
          if debug:
            print(f'### Updated FITS filename {result} with '
                  f'content hash {content_hash}')

        return FileArtifact(result, content_hash)

      write_mode = (WriteMode.MEMORY_AND_DISK
                    if keep_in_memory else WriteMode.DISK_ONLY)
      context.set(key, result, write_mode)
      if debug:
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
      if debug:
        print(f'### key {key}')

      if _result_exists(content_hash):
        if debug:
          print(f'### Not running fn {fn.__name__} producing content '
                f'hash {content_hash} because result exists')
        return CacheArtifact(key)
      else:
        if debug:
          print(f'### Running fn {fn.__name__} producing content '
                f'hash {content_hash} because result does not exist')
        result = fn(*args, **kwargs)
        return _store_result(result, content_hash)

    return wrapper

  return decorator
