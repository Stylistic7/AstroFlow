"""The app driver holds a macOS power assertion for its own lifetime.

Regression for 2026-09-08/09: the MBP worker slept three times in one night
and was evicted by the GCS each time. Nothing in the app held an assertion,
so staying awake depended on being on AC; a charge limiter's discharge cycle
put the machine on battery (`sleep 1`) and it went down mid-integration at
880% CPU. macOS idle sleep does not care how busy the cores are. The sleeps
also dropped the SMB mounts, after which every task on that node failed with
`PermissionError: [Errno 13] /Volumes/AstroProcessed`.
"""
import os
import subprocess

import main


class _FakeProc:
    pid = 4242


def test_holds_an_assertion_that_dies_with_us(monkeypatch):
    calls = []
    monkeypatch.setattr(main.sys, "platform", "darwin")
    monkeypatch.setattr(main.subprocess, "Popen",
                        lambda cmd, **kw: calls.append((cmd, kw)) or _FakeProc())

    main._prevent_idle_sleep()

    (cmd, kw), = calls
    # `-i` = idle sleep only: a headless node's DISPLAY should still blank.
    # `-w <our pid>` = caffeinate exits when we do, so it can never leak.
    assert cmd == ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]
    assert "-d" not in cmd
    assert kw["stdin"] is subprocess.DEVNULL


def test_no_op_off_darwin(monkeypatch):
    def _spawned(*a, **k):
        raise AssertionError("spawned caffeinate off darwin")

    monkeypatch.setattr(main.sys, "platform", "linux")
    monkeypatch.setattr(main.subprocess, "Popen", _spawned)
    main._prevent_idle_sleep()


def test_a_spawn_failure_never_stops_the_app(monkeypatch):
    """Best-effort by design — losing the assertion is worth a warning, not
    a boot failure."""
    monkeypatch.setattr(main.sys, "platform", "darwin")

    def _boom(*a, **k):
        raise OSError("no caffeinate here")

    monkeypatch.setattr(main.subprocess, "Popen", _boom)
    main._prevent_idle_sleep()          # must not raise
