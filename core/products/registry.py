from enum import Enum, auto
from pathlib import Path
from typing import Callable, Any, Type, Optional, Union

from file_io.types import ContentHashable
from previews.types import QualityLevel, Implementation
from context import context


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


type FilenameFn = Union[str, Callable[..., Path]]

_registry: dict[
    MediaType, dict[tuple[Type, QualityLevel, Implementation], Callable[..., Path]]
] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}

_filename_fns: dict[
    MediaType, dict[tuple[Type, QualityLevel, Implementation], Callable[..., Path]]
] = {
    MediaType.IMAGE: {},
    MediaType.VIDEO: {},
}

_producers_registered = False


def _ensure_registered():
    global _producers_registered
    if not _producers_registered:
        _producers_registered = True
        from products import ensure_producers_registered
        ensure_producers_registered()


def register_producer(
    media_type: MediaType,
    object_type: type,
    level: QualityLevel,
    func: Callable[..., Path],
    filename_fn: "FilenameFn | None" = None,
    implementation: Implementation = Implementation.PIXINSIGHT,
) -> None:
    key = (object_type, level, implementation)
    _registry[media_type][key] = func
    if filename_fn is not None:
        _filename_fns[media_type][key] = filename_fn


def get_producer(
    media_type: MediaType, object_type: type, level: QualityLevel,
    implementation: Implementation = Implementation.PIXINSIGHT,
) -> Optional[Callable[..., Path]]:
    _ensure_registered()
    return _registry[media_type].get((object_type, level, implementation))


def get_filename_fn(
    media_type: MediaType, object_type: type, level: QualityLevel,
    implementation: Implementation = Implementation.PIXINSIGHT,
) -> Optional[Callable[..., Path]]:
    _ensure_registered()
    return _filename_fns[media_type].get((object_type, level, implementation))


def available_levels(
    media_type: MediaType, object_type: type,
    implementation: Implementation = Implementation.PIXINSIGHT,
) -> set[QualityLevel]:
    return {
        level
        for (otype, level, impl) in _registry[media_type]
        if otype is object_type and impl == implementation
    }


def clear_registry(media_type: MediaType) -> None:
    _registry[media_type].clear()
    _filename_fns[media_type].clear()


def _mark_valid(object_type: type, level: QualityLevel, implementation: Implementation, args: tuple) -> None:
    """Mark a product as valid after its producer completes."""
    # The first argument to a producer is always the primary object
    # (Stack, Mosaic, etc.) which has stable_hash.
    obj = args[0] if args else None
    if obj is None or not hasattr(obj, "stable_hash"):
        return
    key = (object_type.__name__.lower(), obj.stable_hash, level.value, implementation.value)
    try:
        from context import context
        context.or_equals("validity", {key: True})
    except Exception:
        pass  # Don't let validity tracking break the producer


def invalidate_object(object_type: str, object_id: str, implementation: str = None) -> None:
    """Mark all quality levels of an object as invalid."""
    from context import context
    implementations = [implementation] if implementation else [i.value for i in Implementation]
    updates = {}
    for level in QualityLevel:
        for impl in implementations:
            updates[(object_type, object_id, level.value, impl)] = False
    context.or_equals("validity", updates)


def is_object_valid(object_type: str, object_id: str, level: str, implementation: str = None) -> bool:
    """Query validity, deriving parent validity from children.

    Returns True if the product is up to date, False otherwise
    (stale or never produced).
    """
    from context import context
    if implementation is None:
        implementation = context.settings.implementation.value

    if object_type == "mosaic":
        mosaics = context.mosaics or {}
        if object_id not in mosaics:
            return False
        mosaic = mosaics[object_id].get()
        return all(is_object_valid("target", tid, level, implementation) for tid in mosaic.target_ids)

    if object_type == "target":
        targets = context.targets or {}
        if object_id not in targets:
            return False
        target = targets[object_id].get()
        return all(is_object_valid("stack", sid, level, implementation) for sid in target.stack_ids)

    # Leaf: valid only if explicitly marked True.
    validity = context.get("validity", {})
    return validity.get((object_type, object_id, level, implementation)) is True


def fully_resolve(
    object_type: type, level: QualityLevel, filename_fn: FilenameFn,
    implementation: Implementation = Implementation.PIXINSIGHT,
):
    def inner(*args, **kwargs):
        filename = filename_fn(*args, **kwargs)
        assert isinstance(filename, ContentHashable)
        return filename.resolve_at(
            context.output_dir / str(object_type.__name__) / str(level) / str(implementation)
        )

    return inner


def _bind_producer(
    media_type: MediaType,
    object_type: type,
    implementation: Implementation | list[Implementation],
    levels: list[QualityLevel],
    filename_fn: FilenameFn | None,
):
    """Decorator that registers a function as a producer for the given
    (object_type, level, implementation) combinations.

    Usage::

        @node(...)
        @makes_image(Stack, PIXINSIGHT, [MASTER, DOWNSCALED_MASTER], stack_filename)
        def stack_files(stack, config): ...
    """
    implementations = implementation if isinstance(implementation, list) else [implementation]

    def decorator(func: Callable[..., Path]):
        # Decoration order matters: @makes_image / @makes_video must
        # be applied *inside* @node so that we wrap the raw function
        # (not the Ray-task wrapper). cached_decorator stamps
        # `_is_task = True` on the wrapper it returns, so spotting
        # that attribute on `func` means the user wrote the decorators
        # in the wrong order.
        if getattr(func, '_is_task', False):
            deco = "makes_image" if media_type == MediaType.IMAGE else "makes_video"
            raise TypeError(
                f"@{deco} must be applied inside @node, not outside. "
                f"Got @{deco}({object_type.__name__}) wrapping "
                f"a Ray task. Correct order:\n"
                f"    @node(...)\n"
                f"    @{deco}(...)\n"
                f"    def {getattr(func, '__name__', '...')}(...): ..."
            )

        # Accumulate validity entries (shared across stacked decorators
        # for polymorphic producers like stack_files).
        if not hasattr(func, '_validity_entries'):
            func._validity_entries = []

        # Track registry keys so @node can update the registry to
        # point at the outer (Ray-task) wrapper.
        if not hasattr(func, '_registry_keys'):
            func._registry_keys = []

        for impl in implementations:
            for level in levels:
                resolved = None
                if filename_fn is not None:
                    resolved = fully_resolve(object_type, level, filename_fn, impl)
                    if not hasattr(func, 'filename_fns'):
                        func.filename_fns = {}
                    func.filename_fns[(object_type, level)] = resolved
                    func.filename_fn = resolved  # convenience alias (last wins)

                func._validity_entries.append((object_type, level, impl))
                func._registry_keys.append((media_type, object_type, level, impl))

        # Wrap the function once. Subsequent decorator calls (stacked
        # @makes_image decorators for polymorphic producers) skip the
        # wrap and just append to the closure-captured `entries` list.
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
                    for obj_type, lvl, impl in entries:
                        _mark_valid(obj_type, lvl, impl, args)
                    discard_sidecar(result)
                return result

            wrapped._validity_entries = entries
            wrapped._makes_image_wrapped = True
            wrapped._registry_keys = func._registry_keys
            if hasattr(inner_func, 'filename_fns'):
                wrapped.filename_fns = inner_func.filename_fns
            if hasattr(inner_func, 'filename_fn'):
                wrapped.filename_fn = inner_func.filename_fn

        for impl in implementations:
            for level in levels:
                resolved = None
                if filename_fn is not None:
                    resolved = fully_resolve(object_type, level, filename_fn, impl)
                register_producer(
                    media_type, object_type, level, wrapped,
                    filename_fn=resolved,
                    implementation=impl,
                )

        return wrapped

    return decorator


def makes_image(
    object_type: type, implementation: Implementation,
    levels: list[QualityLevel], filename_fn: FilenameFn | None = None,
):
    return _bind_producer(MediaType.IMAGE, object_type, implementation, levels, filename_fn)


def makes_video(
    object_type: type, implementation: Implementation,
    levels: list[QualityLevel], filename_fn: FilenameFn | None = None,
):
    return _bind_producer(MediaType.VIDEO, object_type, implementation, levels, filename_fn)
