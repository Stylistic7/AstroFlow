import structlog
import functools


class LogContainer(list):
    pass


def capture_logs(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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
            wrapper_class=structlog.make_filtering_bound_logger(20),
        )

        try:
            result = func(*args, **kwargs)
        finally:
            structlog.reset_defaults()

        return LogContainer(captured_logs), result

    return wrapper
