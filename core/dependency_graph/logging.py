import structlog
import functools
from collections.abc import Callable
from typing import Any
from beartype import beartype


class LogContainer(list):
    pass


def make_capture_logs_decorator(cache):
    """Create a @capture_logs decorator that stores logs in the given cache.

    Logs are stored at ("logs", function_name, call_hash) where call_hash
    is derived from the function arguments.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            captured_logs = []

            def log_capture_processor(_, __, event_dict):
                captured_logs.append(event_dict)
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
