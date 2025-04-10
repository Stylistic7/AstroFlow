from typing import Callable, Any, Type, Optional, Union
from pathlib import Path

_registry: dict[tuple[Type, Any], Callable[..., Path]] = {}


def register(object_type: type, level, func: Callable[..., Path]) -> None:
  key = (object_type, level)
  if key in _registry:
    raise ValueError(f"Duplicate registration for {key}")
  _registry[key] = func


def get_image_producer(object_type: type, level) -> Callable[..., Path] | None:
  return _registry.get((object_type, level))


def available_image_levels(object_type: type) -> set:
  return {lvl for (otype, lvl) in _registry if otype is object_type}


def all_registered() -> dict[tuple[Type, Any], Callable[..., Path]]:
  return dict(_registry)  # return a copy


def clear_registry() -> None:
  _registry.clear()


def path_template(template: str) -> Callable:
  return lambda obj: Path(template.format(**vars(obj)))


def register_image_producer(
    object_type: type,
    level,
    filename_fn: Optional[Union[str, Callable[..., Path]]] = None):

  if isinstance(filename_fn, str):
    filename_fn = path_template(filename_fn)

  def decorator(func: Callable[..., Path]):
    if filename_fn is not None:
      func.filename_fn = filename_fn

    register(object_type, level, func)

    return func

  return decorator
