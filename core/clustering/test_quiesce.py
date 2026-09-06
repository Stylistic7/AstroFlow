import datetime
from types import SimpleNamespace

from clustering.quiesce import admit_quiesced_frames, is_quiesced

UTC = datetime.timezone.utc
GATE = datetime.timedelta(minutes=15)
RUN_GAP = datetime.timedelta(hours=8)


def _meta(times: dict[str, datetime.datetime]):
    return {h: SimpleNamespace(header={"DATE-OBS": t}) for h, t in times.items()}


def test_hot_run_stays_out():
    """A run whose newest frame rides the frontier is still capturing."""
    t0 = datetime.datetime(2026, 9, 5, 22, 0, tzinfo=UTC)
    times = {f"f{i}": t0 + datetime.timedelta(minutes=5 * i) for i in range(4)}
    frontier = max(times.values())  # newest download IS this run's tail
    admitted = admit_quiesced_frames(
        tuple(times), _meta(times), frontier, GATE, RUN_GAP
    )
    assert admitted == ()


def test_closed_run_admitted_whole():
    """Once the frontier moves a gate past the run, all frames enter."""
    t0 = datetime.datetime(2026, 9, 5, 22, 0, tzinfo=UTC)
    times = {f"f{i}": t0 + datetime.timedelta(minutes=5 * i) for i in range(4)}
    frontier = max(times.values()) + GATE
    admitted = admit_quiesced_frames(
        tuple(times), _meta(times), frontier, GATE, RUN_GAP
    )
    assert set(admitted) == set(times)


def test_closed_run_admitted_hot_run_excluded():
    """Two runs of one stack: the finished one enters, the active one
    waits — membership moves exactly once per completed run."""
    t0 = datetime.datetime(2026, 9, 5, 20, 0, tzinfo=UTC)
    closed = {f"a{i}": t0 + datetime.timedelta(minutes=2 * i) for i in range(5)}
    hot_start = t0 + datetime.timedelta(hours=9)  # > RUN_GAP: separate run
    hot = {f"b{i}": hot_start + datetime.timedelta(minutes=2 * i) for i in range(5)}
    times = {**closed, **hot}
    frontier = max(hot.values())  # downloads current with the hot run
    admitted = admit_quiesced_frames(
        tuple(times), _meta(times), frontier, GATE, RUN_GAP
    )
    assert set(admitted) == set(closed)


def test_final_run_closes_via_frontier_nudge():
    """End of night: the newest run defines the frontier and can never be
    a gate behind it — until the builder nudges the frontier past it."""
    newest = datetime.datetime(2026, 9, 6, 5, 0, tzinfo=UTC)
    assert not is_quiesced(newest, newest, GATE)          # stranded
    assert is_quiesced(newest, newest + GATE, GATE)       # nudge closes it


def test_effective_frontier_stabilizes_quiesced_subsets():
    """A subset with no hot frames gets frontier=None — a STABLE cache
    key across drains. Only subsets with hot frames carry the live
    frontier. (Passing the raw frontier re-keyed all identification on
    every drain: the livelock reborn, 2026-09-05 night.)"""
    from clustering.quiesce import effective_frontier
    frontier = datetime.datetime(2026, 9, 6, 5, 0, tzinfo=UTC)
    old = frontier - datetime.timedelta(hours=2)
    hot = frontier - datetime.timedelta(minutes=5)
    assert effective_frontier(old, frontier, GATE) is None
    assert effective_frontier(hot, frontier, GATE) == frontier
    assert effective_frontier(None, frontier, GATE) is None
    assert effective_frontier(old, None, GATE) is None


def test_no_frontier_admits_everything():
    t0 = datetime.datetime(2026, 9, 5, 22, 0, tzinfo=UTC)
    times = {f"f{i}": t0 for i in range(3)}
    admitted = admit_quiesced_frames(
        tuple(times), _meta(times), None, GATE, RUN_GAP
    )
    assert set(admitted) == set(times)


def test_naive_and_aware_datetimes_compare():
    """Legacy frames can carry naive DATE-OBS; the gate must not crash."""
    naive = datetime.datetime(2026, 9, 5, 22, 0)
    aware_frontier = datetime.datetime(2026, 9, 5, 23, 0, tzinfo=UTC)
    assert is_quiesced(naive, aware_frontier, GATE)
