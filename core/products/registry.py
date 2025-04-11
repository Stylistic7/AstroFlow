from enum import Enum, auto
from pathlib import Path
from typing import Callable, Any, Type, Optional, Union

from previews.types import QualityLevel


class MediaType(str, Enum):
  IMAGE = "image"
  VIDEO = "video"


_registry: dict[MediaType, dict[tuple[Type, QualityLevel],
                                Callable[..., Path]]] = {
                                    MediaType.IMAGE: {},
                                    MediaType.VIDEO: {},
                                }


def path_template(template: str) -> Callable:
  return lambda obj: Path(template.format(**vars(obj)))


def register_producer(media_type: MediaType, object_type: type,
                      level: QualityLevel, func: Callable[..., Path]) -> None:
  key = (object_type, level)
  if key in _registry[media_type]:
    raise ValueError(f"Duplicate registration for {media_type.value} {key}")
  _registry[media_type][key] = func


def get_producer(media_type: MediaType, object_type: type,
                 level: QualityLevel) -> Optional[Callable[..., Path]]:
  return _registry[media_type].get((object_type, level))


def available_levels(media_type: MediaType,
                     object_type: type) -> set[QualityLevel]:
  return {
      level for (otype, level) in _registry[media_type] if otype is object_type
  }


def clear_registry(media_type: MediaType) -> None:
  _registry[media_type].clear()


class MediaProducerBuilder:

  def __init__(self, media_type: MediaType, object_type: type):
    self.media_type = media_type
    self.object_type = object_type

  def _bind(self,
            level: QualityLevel,
            filename_fn: Optional[Union[str, Callable[..., Path]]] = None):
    if isinstance(filename_fn, str):
      filename_fn = path_template(filename_fn)

    def decorator(func: Callable[..., Path]):
      if filename_fn is not None:
        func.filename_fn = filename_fn
      register_producer(self.media_type, self.object_type, level, func)
      return func

    return decorator

  def as_subframe(self,
                  filename_fn: Optional[Union[str, Callable[...,
                                                            Path]]] = None):
    return self._bind(QualityLevel.SUBFRAME, filename_fn)

  def as_master(self,
                filename_fn: Optional[Union[str, Callable[..., Path]]] = None):
    return self._bind(QualityLevel.MASTER, filename_fn)

  def as_post_processed(self,
                        filename_fn: Optional[Union[str,
                                                    Callable[...,
                                                             Path]]] = None):
    return self._bind(QualityLevel.POST_PROCESSED, filename_fn)


def makes_image(object_type: type) -> MediaProducerBuilder:
  return MediaProducerBuilder(MediaType.IMAGE, object_type)


def makes_video(object_type: type) -> MediaProducerBuilder:
  return MediaProducerBuilder(MediaType.VIDEO, object_type)
