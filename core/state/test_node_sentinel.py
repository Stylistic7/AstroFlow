"""Wedge-detection tests for the per-node sentinel.

Pure state-machine tests: `_pi_snapshot` and the clock are faked, so no
Ray, no PixInsight, no psutil scanning. The actor's undecorated class is
reached through Ray's metadata so nothing here needs a cluster.
"""
import time

from state.node_sentinel import NodeSentinel, _PI_WEDGE_SECONDS


def _bare():
    cls = getattr(NodeSentinel, "__ray_actor_class__", None) or \
        NodeSentinel.__ray_metadata__.modified_class
    return cls()


def test_active_pi_is_never_wedged(monkeypatch):
    s = _bare()
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    cpu = [10.0]
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [(42, cpu[0], 0.0)])
    assert s.pixinsight_health()[0]["wedged"] is False
    # CPU keeps advancing across a long span — never wedged.
    for _ in range(10):
        now[0] += _PI_WEDGE_SECONDS
        cpu[0] += 5.0
        assert s.pixinsight_health()[0]["wedged"] is False


def test_silent_pi_wedges_after_the_window(monkeypatch):
    s = _bare()
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [(42, 500.0, 0.0)])
    s.pixinsight_health()                     # baseline
    now[0] += _PI_WEDGE_SECONDS - 10
    assert s.pixinsight_health()[0]["wedged"] is False, "not yet"
    now[0] += 20
    h = s.pixinsight_health()[0]
    assert h["wedged"] is True and h["inactive_s"] > _PI_WEDGE_SECONDS


def test_idle_pooled_instance_is_never_wedged(monkeypatch):
    """A PI with only startup CPU (no real work) idling past the window is a
    POOL resource, not a wedge — reaping it just forces a cold start."""
    s = _bare()
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [(42, 15.0, 0.0)])
    s.pixinsight_health()
    now[0] += _PI_WEDGE_SECONDS * 10
    assert s.pixinsight_health()[0]["wedged"] is False


def test_scratch_write_counts_as_activity(monkeypatch):
    """An I/O-bound phase burns no CPU but touches scratch — not wedged."""
    s = _bare()
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    writes = [0.0]
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [(42, 10.0, writes[0])])
    s.pixinsight_health()
    now[0] += _PI_WEDGE_SECONDS + 60
    writes[0] = now[0] - 5                     # scratch touched just now
    assert s.pixinsight_health()[0]["wedged"] is False


def test_elapsed_time_alone_is_not_a_signal(monkeypatch):
    """The 2026-08-19 6.6h WBPP was legitimate; hours of ELAPSED time with
    ongoing CPU must never trip the watchdog (big WBPP legit >1h)."""
    s = _bare()
    now = [0.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    cpu = [0.0]
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [(42, cpu[0], 0.0)])
    for _ in range(100):                       # ~8 hours of 5-min steps
        now[0] += 290
        cpu[0] += 1.0
        assert s.pixinsight_health()[0]["wedged"] is False


def test_departed_pid_is_forgotten(monkeypatch):
    s = _bare()
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [(42, 10.0, 0.0)])
    s.pixinsight_health()
    monkeypatch.setattr(s, "_pi_snapshot", lambda: [])
    assert s.pixinsight_health() == []
    assert s._pi_activity == {}


def test_module_imports_stay_lean():
    """The sentinel lives FOREVER on every node, so its process must never
    import the app's heavy module graph (numpy/pyvips/context pull hundreds
    of MB into a worker that otherwise idles at ~66 MB). Measured floor
    2026-08-20: ray+psutil interpreter = 66 MB. This test is the fence: the
    module may import ONLY stdlib + ray at top level; psutil only inside
    methods (loaded on first use, on the node, never in the driver)."""
    import ast, pathlib
    src = pathlib.Path("state/node_sentinel.py").read_text()
    tree = ast.parse(src)
    allowed = {"os", "subprocess", "sys", "time", "ray"}
    top = []
    for node in tree.body:                      # TOP-LEVEL imports only
        if isinstance(node, ast.Import):
            top += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append((node.module or "").split(".")[0])
    bad = [m for m in top if m not in allowed]
    assert not bad, f"heavy/app imports at sentinel module top level: {bad}"
