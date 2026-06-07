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

    Returns one key per identity the producer registers for (a
    @makes_image-decorated function publishes `_identities`). Empty
    list if the function isn't a registered producer — concerns from
    such functions still live in the per-call log, just not surfaced
    per-object.

    The key shape is `('concerns', object_type, identity_key)` and
    the stored value is a dict `{op_name: [concerns...]}` keyed by
    which op wrote each batch. Direct lookup by object replaces the
    earlier op-in-key shape that forced `/api/concerns` to scan the
    full cache to find every op that contributed. Polymorphic
    producers (e.g. stack_files for both StackId and
    CalibrationStackId) emit one key per identity they match — the
    dispatch picks the one whose object_type appears in args, so a
    CalibrationStack run doesn't write under StackId and vice versa.
    """
    identities = getattr(func, "_identities", None)
    if not identities:
        return []

    from products.registry import _extract_identity

    keys: list[tuple] = []
    for identity_cls in identities:
        object_type = identity_cls.object_type
        # Only include entries whose object_type matches an actual arg
        # (a polymorphic producer with both Stack and CalibrationStack
        # registrations only fires for the one it's actually processing).
        if not any(isinstance(a, object_type) for a in args):
            continue
        try:
            obj_id = _extract_identity(identity_cls, func, args, kwargs)
        except Exception:
            continue
        keys.append((
            "concerns",
            object_type.__name__,
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

                # Always overwrite this op's slot in the per-object
                # concerns dict (even with []) so a successful re-run
                # that emitted no concerns clears the previous run's.
                # Skip if not a registered producer.
                concerns = [e for e in captured_logs if e.get("_concern")]
                op_name = func.__name__
                for key in _concerns_keys_for(func, args, kwargs):
                    with cache.transact():
                        existing = cache.get(key, {})
                        if not isinstance(existing, dict):
                            existing = {}
                        existing[op_name] = concerns
                        cache[key] = existing

            return result

        return wrapper

    return decorator
