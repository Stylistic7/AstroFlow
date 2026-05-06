import structlog
import functools
from collections.abc import Callable
from typing import Any
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


def make_capture_logs_decorator(cache):
    """Create a @capture_logs decorator that stores logs in the given cache.

    Logs are stored at ("logs", function_name, call_hash) where call_hash
    is derived from the function arguments. The first event of each
    captured task also gets echoed to stderr for real-time visibility.
    """
    def decorator(func: Callable) -> Callable:
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
                    call_hash = hash_call(fn_hash, *args, **kwargs)
                    cache[("logs", func.__name__, call_hash)] = captured_logs

            return result

        return wrapper

    return decorator
