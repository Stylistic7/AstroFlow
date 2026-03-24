import structlog
import functools
from collections.abc import Callable
from typing import Any
from beartype import beartype


class LogContainer(list):
    pass


@beartype
def capture_logs(func: Callable) -> Callable:
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
        except Exception as e:
            return LogContainer(captured_logs), e
        finally:
            structlog.reset_defaults()

        return LogContainer(captured_logs), result

    return wrapper
