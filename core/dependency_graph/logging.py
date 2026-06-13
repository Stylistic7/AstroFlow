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

    The producer's first positional arg is the entity. We key under
    `('concerns', type(entity).__name__, entity.stable_hash)`. Mosaic
    producers carry a `filter` second arg we splice in to match the
    `/api/concerns/<mosaic>` lookup shape.

    Returns one key (the producer's entity key). Empty list if no
    positional args or no `stable_hash` on the first arg — we don't
    have an object to attribute concerns to.

    The stored value is `{op_name: [concerns...]}` so distinct ops
    that contribute to the same object don't overwrite each other.
    """
    if not args:
        return []
    entity = args[0]
    stable_hash = getattr(entity, "stable_hash", None)
    if not stable_hash:
        return []
    type_name = type(entity).__name__
    # Mosaic-keyed concerns include the filter (`/api/concerns/mosaic`
    # passes `?filter=H`; that filter is part of the key shape the
    # endpoint constructs). Detect by the second positional arg being
    # a string — the per-filter convention.
    if type_name == "Mosaic" and len(args) >= 2 and isinstance(args[1], str):
        identity_key = f"{stable_hash}_{args[1]}"
    else:
        identity_key = stable_hash
    return [("concerns", type_name, identity_key)]


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
