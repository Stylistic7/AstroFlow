import os
from functools import wraps

from context import context
from file_io.generic import get_content_hash


def saves_artifacts(artifacts_fn):

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)

      if type(artifacts) is dict:
        for content_hash, filename in artifacts.items():
          filename = os.path.join(context.output_dir, filename)
          if not os.path.exists(filename) or get_content_hash(
              filename) != content_hash:
            return True

        return False
      elif type(artifacts) is list:
        return not all(context.contains(k) for k in artifacts)

    @wraps(fn)
    def wrapper(*args, **kwargs):
      return fn(*args, **kwargs)

    wrapper.should_run = should_run

    if hasattr(fn, 'remote'):
      wrapper.remote = fn.remote

    return wrapper

  return decorator
