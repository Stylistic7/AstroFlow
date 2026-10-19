"""Unit tests for GunicornManager's boot-failure healing.

The numpy longdouble import race kills 1-2 of 5 simultaneously-spawned
servers on most starts (macOS, numpy 2.x); gunicorn treats a worker boot
failure as fatal to its master, so a raced server used to die silently and
serve nothing until noticed. `_heal_boot_failures` is the recovery. These
tests fake `_spawn`, so no gunicorn, no Ray, no sockets.
"""
import time

from state.gunicorn_control import GunicornConfig, GunicornManager


class _FakeProc:
    """poll() returns None (alive) or an exit code, per the script given."""
    def __init__(self, polls):
        self._polls = list(polls)
        self.returncode = None

    def poll(self):
        v = self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]
        if v is not None:
            self.returncode = v
        return v


def _manager(monkeypatch, spawn_results, num_servers):
    cfg = GunicornConfig(
        name="t", config_file=None, module="m:app", worker_type="sync",
        num_servers=num_servers, start_port=9900,
        num_workers=1, timeout=300,
    )
    m = GunicornManager(cfg)
    it = iter(spawn_results)

    def fake_spawn(port):
        return next(it)

    monkeypatch.setattr(m, "_spawn", fake_spawn)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return m


def test_healthy_fleet_spawns_once(monkeypatch):
    procs = [_FakeProc([None]) for _ in range(3)]
    m = _manager(monkeypatch, procs, num_servers=3)
    m.start()
    assert m.processes == procs, "no respawn when everything boots"


def test_boot_failure_is_respawned(monkeypatch):
    dead = _FakeProc([3])                 # loses the import race, exits 3
    replacement = _FakeProc([None])       # the lone retry succeeds (3/3 observed)
    healthy = _FakeProc([None])
    m = _manager(monkeypatch, [dead, healthy, replacement], num_servers=2)
    m.start()
    assert m.processes[0] is replacement
    assert m.processes[1] is healthy


def test_respawn_rounds_are_bounded(monkeypatch):
    # A port that dies every round must not loop forever — 1 spawn + 3 rounds.
    always_dead = [_FakeProc([3]) for _ in range(4)]
    m = _manager(monkeypatch, always_dead, num_servers=1)
    m.start()
    assert m.processes[0] is always_dead[3], "exactly MAX_RESPAWN_ROUNDS retries"


def _config(**overrides):
    return GunicornConfig(**{
        "name": "t", "config_file": "web_server/backend/gunicorn_config.py", "module": "m:app",
        "worker_type": "gevent", "num_servers": 1, "start_port": 8001, "num_workers": 1,
        "timeout": 0, **overrides})


def test_posix_runs_gunicorn_and_windows_runs_waitress(monkeypatch):
    import sys
    cfg = _config(threads=64, on_start="web_server.backend.announcer:start_event_loop")
    monkeypatch.setattr(sys, "platform", "darwin")
    posix = GunicornManager(cfg)._command(8001)
    assert posix[1:3] == ["-m", "gunicorn"] and "--threads" in posix and "64" not in posix
    assert "-c" in posix, "gunicorn's config file starts the announcer there"
    monkeypatch.setattr(sys, "platform", "win32")
    windows = GunicornManager(cfg)._command(8001)
    assert windows[1:4] == ["-m", "web_server.backend.serve", "m:app"]
    assert windows[4:] == ["--port", "8001", "--threads", "64",
                           "--on-start", "web_server.backend.announcer:start_event_loop"]
    assert "--on-start" not in GunicornManager(_config())._command(8000)
