import os
from functools import wraps

from context import context
from file_io.generic import get_content_hash


def saves_files(artifacts_fn):

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)
      for content_hash, filename in artifacts.items():
        filename = os.path.join(context.output_dir, filename)
        if not os.path.exists(filename) or get_content_hash(
            filename) != content_hash:
          return True

      return False

    @wraps(fn)
    def wrapper(*args, **kwargs):
      assert False, "Do not call this function directly."

    wrapper.should_run = should_run
    wrapper.remote = fn.remote

    return wrapper

  return decorator


def saves_cache_keys(keys_fn):

  def decorator(fn):

    def should_run(*args, **kwargs):
      keys = keys_fn(*args, **kwargs)
      return not all(context.contains(k) for k in keys)

    @wraps(fn)
    def wrapper(*args, **kwargs):
      assert False, "Do not call this function directly."

    wrapper.should_run = should_run
    wrapper.remote = fn.remote

    return wrapper

  return decorator
