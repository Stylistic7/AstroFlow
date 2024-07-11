import os
from functools import wraps

from artifact import Artifact, artifact_exists
from context import context
from file_io.generic import get_content_hash


def saves_artifacts(artifacts_fn, in_test=False, debug=True):

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)
      assert type(artifacts) is list

      for artifact in artifacts:
        assert isinstance(
            artifact, Artifact), f'{fn.__name__} returned type {type(artifact)}'

        if not artifact_exists(artifact):
          print(f'### should_run fn {fn.__name__} because artifact '
                f'{artifact} does not exist')
          return True

      if debug:
        print(f'### not running {fn.__name__} artifacts {artifacts}')

      return False

    @wraps(fn)
    def wrapper(*args, **kwargs):
      if should_run(*args, **kwargs):
        return fn(*args, **kwargs)

    if in_test:
      wrapper.should_run = should_run

    return wrapper

  return decorator
