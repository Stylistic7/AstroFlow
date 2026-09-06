"""Run-quiesce gating for identification (design: Stylistic7, 2026-09-05).

A run still being captured must not enter a stack: every new frame would
re-key the stack and cascade recomputation on each download drain (the
overnight livelock). A run is QUIESCED when the library FRONTIER — the
newest capture time anywhere on disk, nudged past the end of a finished
session by the impure builder layer — is at least the gate window past
the run's newest frame. Identification admits only quiesced runs, so a
hot run stays invisible until a later drain proves it complete, and a
drain that closes no run re-derives identical stacks (all cache hits)
and returns to frontier work within minutes.

Pure functions of (metadata, frontier, gate): no clock in here. The one
wall-clock judgment — "the session is over, close the final runs" —
lives in the @ray.remote builder that computes the frontier.
"""
import datetime
from collections.abc import Mapping

from beartype import beartype

from clustering.continuous import cluster_continuous


def _utc(t: datetime.datetime) -> datetime.datetime:
    """Comparisons need uniform awareness; legacy frames can be naive."""
    return t if t.tzinfo is not None else t.replace(tzinfo=datetime.timezone.utc)


@beartype
def is_quiesced(
    newest: datetime.datetime,
    frontier: datetime.datetime | None,
    gate: datetime.timedelta,
) -> bool:
    if frontier is None:
        return True
    return _utc(newest) <= _utc(frontier) - gate


@beartype
def effective_frontier(
    subset_newest: datetime.datetime | None,
    frontier: datetime.datetime | None,
    gate: datetime.timedelta,
) -> datetime.datetime | None:
    """The frontier a task actually needs in its cache key.

    Once everything a task can see is provably quiesced (the subset's
    newest frame is a full gate behind the frontier), the gate cannot
    change that task's output — so pass None and keep its key STABLE
    across drains. Passing the raw, ever-advancing frontier re-keyed
    every identify task on every drain and re-ran full identification
    each 15-minute cycle (2026-09-05 night — the livelock reborn).
    Only tasks with hot frames carry the live frontier, re-keying
    exactly until their runs close."""
    if frontier is None or subset_newest is None:
        return None
    if _utc(frontier) - _utc(subset_newest) >= gate:
        return None
    return frontier


@beartype
def admit_quiesced_frames(
    frames: tuple[str, ...],
    metadata: Mapping,
    frontier: datetime.datetime | None,
    gate: datetime.timedelta,
    max_run_gap: datetime.timedelta,
) -> tuple[str, ...]:
    """Drop the frames of runs that have not quiesced.

    Frames are clustered into runs with the SAME gap rule stack formation
    uses, so the admitted set is always a whole number of runs — a hot
    run never leaks in piecewise, which is what kept re-keying stacks on
    every drain.
    """
    if frontier is None or not frames:
        return frames
    items = [(h, _utc(metadata[h].header["DATE-OBS"])) for h in frames]
    admitted: list[str] = []
    for run in cluster_continuous(items, epsilon=max_run_gap):
        newest = max(t for _, t in run)
        if is_quiesced(newest, frontier, gate):
            admitted.extend(h for h, _ in run)
    return tuple(admitted)
