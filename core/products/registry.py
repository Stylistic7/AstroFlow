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


def _mark_valid(object_type: type, level: QualityLevel, args: tuple) -> None:
    """Mark a product as valid after its producer completes."""
    # The first argument to a producer is always the primary object
    # (Stack, Mosaic, etc.) which has stable_hash.
    obj = args[0] if args else None
    if obj is None or not hasattr(obj, "stable_hash"):
        return
    key = (object_type.__name__.lower(), obj.stable_hash, level.value)
    try:
        from context import context
        context.or_equals("validity", {key: True})
    except Exception:
        pass  # Don't let validity tracking break the producer


def invalidate_object(object_type: str, object_id: str) -> None:
    """Mark all quality levels of an object as invalid."""
    from context import context
    updates = {}
    for level in QualityLevel:
        updates[(object_type, object_id, level.value)] = False
    context.or_equals("validity", updates)


def is_object_valid(object_type: str, object_id: str, level: str) -> bool:
    """Query validity, deriving parent validity from children.

    Returns True if the product is up to date, False otherwise
    (stale or never produced).
    """
    from context import context

    if object_type == "mosaic":
        mosaics = context.mosaics or {}
        if object_id not in mosaics:
            return False
        mosaic = mosaics[object_id].get()
        return all(is_object_valid("target", tid, level) for tid in mosaic.target_ids)

    if object_type == "target":
        targets = context.targets or {}
        if object_id not in targets:
            return False
        target = targets[object_id].get()
        return all(is_object_valid("stack", sid, level) for sid in target.stack_ids)

    # Leaf: valid only if explicitly marked True.
    validity = context.get("validity", {})
    return validity.get((object_type, object_id, level)) is True


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
            # Decoration order matters: @makes_image / @makes_video must
            # be applied *inside* @node so that we wrap the raw function
            # (not the Ray-task wrapper). cached_decorator stamps
            # `_is_task = True` on the wrapper it returns, so spotting
            # that attribute on `func` means the user wrote the decorators
            # in the wrong order.
            if getattr(func, '_is_task', False):
                deco = (
                    "makes_image" if self.media_type == MediaType.IMAGE
                    else "makes_video"
                )
                raise TypeError(
                    f"@{deco} must be applied inside @node, not outside. "
                    f"Got @{deco}({self.object_type.__name__}) wrapping "
                    f"a Ray task. Correct order:\n"
                    f"    @node(...)\n"
                    f"    @{deco}(...).for_levels(...)\n"
                    f"    def {getattr(func, '__name__', '...')}(...): ..."
                )

            resolved = None
            if filename_fn is not None:
                resolved = fully_resolve(self.object_type, level, filename_fn)
                # Accumulate per-(object_type, level) filename_fns so stacking
                # multiple @makes_image decorators on one function works.
                if not hasattr(func, 'filename_fns'):
                    func.filename_fns = {}
                func.filename_fns[(self.object_type, level)] = resolved
                func.filename_fn = resolved  # convenience alias (last wins)

            # Accumulate validity entries (shared across stacked
            # @makes_image decorators via the same list reference).
            if not hasattr(func, '_validity_entries'):
                func._validity_entries = []
            func._validity_entries.append((self.object_type, level))

            # Wrap the function once. Subsequent _bind calls (multiple
            # levels in for_levels, or stacked @makes_image decorators
            # for polymorphic producers like stack_files) skip the wrap
            # and just append to the closure-captured `entries` list,
            # which the wrap iterates at call time.
            if getattr(func, '_makes_image_wrapped', False):
                wrapped = func
            else:
                from functools import wraps
                from file_io.sidecar import discard_sidecar
                inner_func = func
                entries = func._validity_entries  # closure-captured (shared)

                @wraps(inner_func)
                def wrapped(*args, **kwargs):
                    result = inner_func(*args, **kwargs)
                    if result is not None:
                        for obj_type, lvl in entries:
                            _mark_valid(obj_type, lvl, args)
                        # The new image obsoletes any image_metadata /
                        # plate_solution in the sidecar (computed from
                        # the previous file content). Discard it so the
                        # next compute_image_metadata call re-derives.
                        discard_sidecar(result)
                    return result

                wrapped._validity_entries = entries
                wrapped._makes_image_wrapped = True
                if hasattr(inner_func, 'filename_fns'):
                    wrapped.filename_fns = inner_func.filename_fns
                if hasattr(inner_func, 'filename_fn'):
                    wrapped.filename_fn = inner_func.filename_fn

            register_producer(
                self.media_type, self.object_type, level, wrapped,
                filename_fn=resolved,
            )
            return wrapped

        return decorator

    def for_levels(
        self,
        levels: list[QualityLevel] | tuple[QualityLevel, ...],
        filename_fn: FilenameFn | None = None,
    ):
        """Register a producer for one or more quality levels.

        Usage:
            @node(...)
            @makes_image(Stack).for_levels([MASTER, DOWNSCALED_MASTER], stack_filename)
            def stack_files(stack, config): ...
        """
        def decorator(func: Callable[..., Path]):
            for level in levels:
                func = self._bind(level, filename_fn)(func)
            return func
        return decorator


def makes_image(object_type: type) -> MediaProducerBuilder:
    return MediaProducerBuilder(MediaType.IMAGE, object_type)


def makes_video(object_type: type) -> MediaProducerBuilder:
    return MediaProducerBuilder(MediaType.VIDEO, object_type)
