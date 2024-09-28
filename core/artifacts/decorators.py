import os
from functools import wraps
from typing import Callable, TypeVar, Any, ParamSpec

from artifacts import Artifact, FileArtifact, CacheArtifact, artifact_exists

ArtifactList = list[Artifact]
ArtifactFn = Callable[..., list[Artifact]]

P = ParamSpec("P")


def should_run(artifacts_fn: ArtifactFn, *args: Any, **kwargs: Any) -> bool:
  artifacts: list[Artifact] = artifacts_fn(*args, **kwargs)
  assert type(
      artifacts
  ) is list, f'artifacts_fn {artifacts_fn.__name__} returned non-list {artifacts}'

  for artifact in artifacts:
    assert isinstance(artifact, (FileArtifact, CacheArtifact))

    if not artifact_exists(artifact):
      return True

  return False


def saves_artifacts(
    artifacts_fn: ArtifactFn
) -> Callable[[Callable[P, ArtifactList]], Callable[P, ArtifactList]]:
  assert not hasattr(artifacts_fn, 'remote')

  def decorator(fn: Callable[P, ArtifactList]) -> Callable[P, ArtifactList]:

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> ArtifactList:
      if should_run(artifacts_fn, *args, **kwargs):
        return fn(*args, **kwargs)
      else:
        return artifacts_fn(*args, **kwargs)

    return wrapper

  return decorator
