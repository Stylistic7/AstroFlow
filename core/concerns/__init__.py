"""User-facing "concerns" raised by background tasks.

A concern is a structured event that the user should review on the
relevant object's detail page — e.g., "WBPP integrated only 35 of 87
frames", "plate-solving failed for the reference frame", "stack has too
few frames after applying decisions". Concerns are non-fatal: they
don't abort the task, they flag a result for review.

Mechanics: concerns are emitted via the same structlog pipeline as
every other log event, but tagged with a `_concern` field. The
surrounding `@capture_logs` decorator picks them up alongside the rest
of the call's log entry, and additionally side-writes them to a
per-object cache key the UI reads from. Re-running the producer
overwrites the per-object key, so concerns track the current run's
state without separate ack/dismiss machinery.

Producers don't need to know any of this — call `concerns.warn(...)`
or `concerns.error(...)` from inside a `@node`-decorated function and
the rest is automatic.
"""
import structlog

_log = structlog.get_logger()


def warn(title: str, detail: str = "", **fields) -> None:
    """Flag a non-fatal anomaly that the user should review.

    `title` is the one-line summary shown on the badge and as the row
    headline in the dialog. `detail` is the longer explanation shown
    when the row is expanded. Additional structured fields (frame
    counts, ratios, filenames, etc.) are preserved on the event for
    debugging and may surface in the dialog later.
    """
    _log.warning(title, _concern="warning", detail=detail, **fields)


def error(title: str, detail: str = "", **fields) -> None:
    """Flag a result that's likely wrong, even though the task didn't raise.

    Use this for cases where the task technically completed but the
    output isn't trustworthy — e.g., integration time off by >50%,
    plate-solving failed on most frames, etc.
    """
    _log.error(title, _concern="error", detail=detail, **fields)
