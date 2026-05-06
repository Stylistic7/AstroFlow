import structlog
import functools
from collections.abc import Callable
from typing import Any, Optional
from beartype import beartype


class LogContainer(list):
    pass


def _print_first_line(fn_name: str, event_dict: dict) -> None:
    """Echo the first structlog event of a captured task to stdout.

    Heavyweight `@capture_logs`-wrapped functions tend to open with a
    `log.info(...)` describing what they're about to do (which stack,
    which mosaic, which target). Surfacing that single line gives
    real-time visibility into "what's running right now" without
    drowning the terminal in every subsequent event from every task.
    Ray forwards stdout to the driver, so the user sees it directly.
    """
    msg = event_dict.get("event", "")
    extras = " ".join(f"{k}={v!r}" for k, v in event_dict.items() if k != "event")
    suffix = f" {extras}" if extras else ""
    print(f"[{fn_name}] {msg}{suffix}", flush=True)


def _concerns_keys_for(func: Callable, args: tuple, kwargs: dict) -> list[tuple]:
    """Per-object cache keys to side-write concerns under.

    Returns one key per (object_type, level, implementation) registration
    on the producer (a `@makes_image`-decorated function publishes the
    `_presentation_entries` attribute). Empty list if the function isn't
    a registered producer — concerns from such functions still live in
    the per-call log, just not surfaced per-object.

    Mirrors the dispatch in `products.registry._generate_presentations`:
    pick the entry whose `object_type` matches an actual arg, extract
    the `ObjectId` via the entry's `identity_cls`, narrow the levels
    via `_detect_level` (when there's a `mode` arg).
    """
    pres_entries = getattr(func, "_presentation_entries", None)
    if not pres_entries:
        return []

    from products.registry import _extract_identity, _detect_level

    keys: list[tuple] = []
    for entry in pres_entries:
        identity_cls = entry.get("identity_cls")
        object_type = entry.get("object_type")
        if identity_cls is None or object_type is None:
            continue
        # Only include entries whose object_type matches a real arg
        # (a polymorphic producer with both Stack and CalibrationStack
        # registrations should only fire for the one it's actually
        # processing).
        if not any(isinstance(a, object_type) for a in args):
            continue
        try:
            obj_id = _extract_identity(identity_cls, func, args, kwargs)
        except Exception:
            continue
        active_levels = _detect_level(func, args, kwargs, entry["levels"])
        for impl in entry["implementations"]:
            for level in active_levels:
                keys.append((
                    "concerns",
                    object_type.__name__,
                    str(level),
                    str(impl),
                    obj_id.cache_key(),
                ))
    return keys


def make_capture_logs_decorator(cache):
    """Create a @capture_logs decorator that stores logs in the given cache.

    Logs go to ("logs", function_name, call_hash). The first event of
    each captured task echoes to stdout for real-time visibility.
    Events tagged with a `_concern` field are also side-written to
    per-object keys ("concerns", object_type, level, impl, object_key)
    so the UI can look them up by the produced object's identity rather
    than by call hash.

    `cache_key_narrow` (optional) narrows args before hashing, matching
    the same hook on `@cached.remote` so the logs land under the same
    key the cache uses.
    """
    def decorator(
        func: Callable = None,
        *,
        cache_key_narrow: Optional[Callable[[tuple, dict], tuple[tuple, dict]]] = None,
    ):
        # Support both @capture_logs and @capture_logs(cache_key_narrow=...)
        if func is None:
            return lambda f: decorator(f, cache_key_narrow=cache_key_narrow)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            captured_logs = []
            first_emitted = False

            def log_capture_processor(_, __, event_dict):
                nonlocal first_emitted
                captured_logs.append(event_dict)
                if not first_emitted:
                    first_emitted = True
                    _print_first_line(func.__name__, event_dict)
                return event_dict

            structlog.configure(
                processors=[
                    log_capture_processor,
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.make_filtering_bound_logger(10),
            )

            try:
                result = func(*args, **kwargs)
            finally:
                structlog.reset_defaults()
                if captured_logs:
                    from hashing import hash_call, hash_source_code
                    fn_hash = hash_source_code(func)
                    if cache_key_narrow is not None:
                        hash_args, hash_kwargs = cache_key_narrow(args, kwargs)
                    else:
                        hash_args, hash_kwargs = args, kwargs
                    call_hash = hash_call(fn_hash, *hash_args, **hash_kwargs)
                    cache[("logs", func.__name__, call_hash)] = captured_logs

                # Always overwrite per-object concerns (even with []) so
                # a successful re-run that emitted no concerns clears
                # the previous run's. Skip if not a registered producer.
                concerns = [e for e in captured_logs if e.get("_concern")]
                for key in _concerns_keys_for(func, args, kwargs):
                    cache[key] = concerns

            return result

        return wrapper

    return decorator
