from enum import Enum, auto
from pathlib import Path
from typing import Callable, Any, Type, Optional, Union

from file_io.types import ContentHashable
from previews.types import QualityLevel
from context import context


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


type FilenameFn = Union[str, Callable[..., Path]]

_registry: dict[
    MediaType, dict[tuple[Type, QualityLevel], Callable[..., Path]]
] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}

_filename_fns: dict[
    MediaType, dict[tuple[Type, QualityLevel], Callable[..., Path]]
] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}


def register_producer(
    media_type: MediaType,
    object_type: type,
    level: QualityLevel,
    func: Callable[..., Path],
    filename_fn: "FilenameFn | None" = None,
) -> None:
    key = (object_type, level)
    _registry[media_type][key] = func
    if filename_fn is not None:
        _filename_fns[media_type][key] = filename_fn


def get_producer(
    media_type: MediaType, object_type: type, level: QualityLevel
) -> Optional[Callable[..., Path]]:
    return _registry[media_type].get((object_type, level))


def get_filename_fn(
    media_type: MediaType, object_type: type, level: QualityLevel
) -> Optional[Callable[..., Path]]:
    return _filename_fns[media_type].get((object_type, level))


def available_levels(
    media_type: MediaType, object_type: type
) -> set[QualityLevel]:
    return {
        level
        for (otype, level) in _registry[media_type]
        if otype is object_type
    }


def clear_registry(media_type: MediaType) -> None:
    _registry[media_type].clear()
    _filename_fns[media_type].clear()


def fully_resolve(
    object_type: type, level: QualityLevel, filename_fn: FilenameFn
):
    def inner(*args, **kwargs):
        filename = filename_fn(*args, **kwargs)
        assert isinstance(filename, ContentHashable)
        return filename.resolve_at(
            context.output_dir / str(object_type.__name__) / str(level)
        )

    return inner


class MediaProducerBuilder:
    def __init__(self, media_type: MediaType, object_type: type):
        self.media_type = media_type
        self.object_type = object_type

    def _bind(self, level: QualityLevel, filename_fn: FilenameFn | None = None):
        def decorator(func: Callable[..., Path]):
            resolved = None
            if filename_fn is not None:
                resolved = fully_resolve(self.object_type, level, filename_fn)
                # Set on the function so it's accessible inside Ray workers
                # (where the process-local registry may not be populated).
                # When multiple levels are registered on the same function,
                # the last decorator wins — use get_filename_fn() for
                # level-specific lookup in the main process.
                func.filename_fn = resolved
            register_producer(
                self.media_type, self.object_type, level, func,
                filename_fn=resolved,
            )
            return func

        return decorator

    def as_subframe(self, filename_fn: FilenameFn | None = None):
        return self._bind(QualityLevel.SUBFRAME, filename_fn)

    def as_downscaled_master(self, filename_fn: FilenameFn | None = None):
        return self._bind(QualityLevel.DOWNSCALED_MASTER, filename_fn)

    def as_master(self, filename_fn: FilenameFn | None = None):
        return self._bind(QualityLevel.MASTER, filename_fn)

    def as_post_processed(self, filename_fn: FilenameFn | None = None):
        return self._bind(QualityLevel.POST_PROCESSED, filename_fn)


def makes_image(object_type: type) -> MediaProducerBuilder:
    return MediaProducerBuilder(MediaType.IMAGE, object_type)


def makes_video(object_type: type) -> MediaProducerBuilder:
    return MediaProducerBuilder(MediaType.VIDEO, object_type)
