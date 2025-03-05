import ray
import inspect
from pathlib import Path
from functools import wraps
import structlog
from typing import Any
import astropy.io.fits as pyfits  # type: ignore
from beartype import beartype

from context import context, WriteMode
from artifacts import FileArtifact, CacheArtifact, artifact_exists
from artifacts.artifact import unpack_artifact
from hashing import hash_list
from dependency_graph import Graph
from dependency_graph.logging import capture_logs
from dependency_graph.classes import canonicalize_node
from dependency_graph.debugging import node_to_string
from profiling import track_performance
from file_io.xisf_file_io import add_xisf_metadata

log = structlog.get_logger()


class ArtifactNotFoundException(Exception):
  pass


class WrongContentHashException(Exception):
  pass


def compute_content_hash(fn, *args, **kwargs) -> str:
  source = inspect.getsource(fn)
  kwargs_list = sorted(kwargs.items())
  return hash_list([source, *args, *kwargs_list])


def delete_cached_result(node):
  node = canonicalize_node(node)
  print(f'### delete_cached_result for node {node_to_string(node)}')
  content_hash = compute_content_hash(node.task._function, *node.args)
  name = node.task._function.__name__
  context.delete((name, content_hash))


def _result_exists(fn, content_hash, debug=False):
  if context.ignore_cached_results:
    print(f'### Node {fn.__name__}: ignoring cached results')
    return False

  key = (fn.__name__, content_hash)
  if not context.contains(key):
    if debug:
      print(f'### Node {fn.__name__}: context does not contain key {key}')
    return False

  result = context[key]
  if isinstance(result, Path):
    if debug:
      print(f'### Node {fn.__name__}: checking that file {result} has '
            f'content hash {content_hash}')
    if not artifact_exists(FileArtifact(result, content_hash)):
      return False

  return True


def _update_header(fn, path, content_hash, debug=False):
  if path.suffix.lower() in ['.xisf']:
    add_xisf_metadata(path, path, {'CONTHASH': content_hash})
    if debug:
      print(f'### Node {fn.__name__}: updated XISF filename {path} '
            f'with content hash {content_hash}')
  elif path.suffix.lower() in ['.fit', '.fits']:
    with pyfits.open(str(path), mode='update', mmap=False) as fits:
      fits[0].header['CONTHASH'] = content_hash
    if debug:
      print(f'### Node {fn.__name__}: updated FITS filename {path} '
            f'with content hash {content_hash}')


def _store_result(fn,
                  result,
                  content_hash,
                  update_header,
                  keep_in_memory,
                  debug=False):
  key = (fn.__name__, content_hash)

  write_mode = (WriteMode.MEMORY_AND_DISK
                if keep_in_memory else WriteMode.DISK_ONLY)

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
  if isinstance(result, Path):
    if update_header:
      _update_header(fn, result, content_hash, debug)

    context.set(key, result, write_mode)
    if debug:
      print(f'### Node {fn.__name__}: stored result {type(result)} '
            f'with key {key} and write mode {write_mode}')
    return FileArtifact(result, content_hash)

  context.set(key, result, write_mode)
  if debug:
    print(f'### Node {fn.__name__}: stored result {type(result)} '
          f'with key {key} and write mode {write_mode}')
  return result


def AstroFlowNode(resources=None,
                  num_cpus=1,
                  memory=None,
                  update_header=True,
                  unpack_artifacts=True,
                  cache_result=False,
                  keep_in_memory=True,
                  debug=False):
  if resources == None:
    resources = {}

  def decorator(fn):
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
                num_cpus=num_cpus,
                memory=memory)
    @capture_logs
    @track_performance
    @wraps(fn)
    @beartype
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

      content_hash = compute_content_hash(fn, *args, **kwargs)
      key = (fn.__name__, content_hash)
      if debug:
        print(f'### Node {fn.__name__}: key {key}')

      if _result_exists(fn, content_hash, debug):
        if debug:
          print(f'### Node {fn.__name__}: not running fn {fn.__name__} '
                f'producing content hash {content_hash} because result exists')
        return context[key]
      else:
        if debug:
          print(f'### Node {fn.__name__}: running fn {fn.__name__} producing '
                f'content hash {content_hash} because result does not exist')
        result = fn(*args, **kwargs)
        return _store_result(fn, result, content_hash, update_header,
                             keep_in_memory, debug)

    return wrapper

  return decorator
