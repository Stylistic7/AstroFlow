import os
from functools import wraps

from artifact import Artifact, artifact_exists
from context import context


def saves_artifacts(artifacts_fn, in_test=False, debug=False):
  assert not hasattr(artifacts_fn, 'remote')

  def decorator(fn):

    def should_run(*args, **kwargs):
      artifacts = artifacts_fn(*args, **kwargs)
      assert type(
          artifacts
      ) is list, f'artifacts_fn {artifacts_fn.__name__} returned non-list {artifacts}'

      for artifact in artifacts:
        assert isinstance(
            artifact, Artifact), f'{fn.__name__} returned type {type(artifact)}'

        if not artifact_exists(artifact):
          if debug:
            print(f'### should_run fn {fn.__name__} because artifact '
                  f'{artifact} does not exist')
          return True

      if debug:
        print(f'### not running {fn.__name__} artifacts {artifacts}')

      return False

    @wraps(fn)
    def wrapper(*args, **kwargs):
      if should_run(*args, **kwargs):
        result = fn(*args, **kwargs)

        assert result is not None, f'{fn.__name__} returned None'
        assert type(
            result) is list, f'{fn.__name__} returned non-list: {result}'
        assert all(isinstance(r, Artifact) for r in result
                  ), f'{fn.__name__} returned list of non-Artifacts: {result}'
        assert all(
            artifact_exists(a) for a in result
        ), f'{fn.__name__} returned artifacts that do not exist: {result}'

        return result
      else:
        return artifacts_fn(*args, **kwargs)

    if in_test:
      wrapper.should_run = should_run

    return wrapper

  return decorator
