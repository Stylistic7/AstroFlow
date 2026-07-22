"""Unit tests for the Ray-control helpers: the bounded `is_ray_connected`
(so a dead remote head can't hang callers) and the local, hang-proof slate
reclaim (`local_raylet_running` / `force_stop_ray`) that replaces the old
`ray status` boot probe.
"""
import subprocess
from types import SimpleNamespace

from state import ray_control as rc


def test_is_ray_connected_times_out_to_false(monkeypatch):
    # A dead/stale remote GCS makes `ray status` hang; we cap it and report
    # "not connected" rather than blocking the caller forever.
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ray status", timeout=k.get("timeout"))
    monkeypatch.setattr(rc.subprocess, "run", _raise)
    assert rc.is_ray_connected(timeout=0.01) is False


def test_is_ray_connected_true_on_autoscaler_status(monkeypatch):
    monkeypatch.setattr(rc.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="Autoscaler status: up"))
    assert rc.is_ray_connected() is True


def test_is_ray_connected_passes_timeout_to_subprocess(monkeypatch):
    seen = {}

    def _run(*a, **k):
        seen["timeout"] = k.get("timeout")
        return SimpleNamespace(stdout="")
    monkeypatch.setattr(rc.subprocess, "run", _run)
    rc.is_ray_connected(timeout=7.5)
    assert seen["timeout"] == 7.5


def test_local_raylet_running_detects_raylet(monkeypatch):
    procs = [SimpleNamespace(info={"name": "python"}),
             SimpleNamespace(info={"name": "raylet"})]
    monkeypatch.setattr(rc.psutil, "process_iter", lambda attrs: iter(procs))
    assert rc.local_raylet_running() is True


def test_local_raylet_running_false_when_absent(monkeypatch):
    procs = [SimpleNamespace(info={"name": "python"}),
             SimpleNamespace(info={"name": "gunicorn"})]
    monkeypatch.setattr(rc.psutil, "process_iter", lambda attrs: iter(procs))
    assert rc.local_raylet_running() is False


def test_force_stop_ray_is_local_and_clears_session(monkeypatch):
    # Reclaim must be local-only: `ray stop --force` + drop the cached session
    # dir. No `ray status` / remote address anywhere (can't hang).
    cmds = []
    monkeypatch.setattr(rc.subprocess, "run",
                        lambda cmd, **k: cmds.append(cmd) or SimpleNamespace(returncode=0))
    removed = []
    monkeypatch.setattr(rc.shutil, "rmtree", lambda p, **k: removed.append(p))
    rc.force_stop_ray()
    assert cmds and cmds[0][-1] == "--force" and "stop" in cmds[0]  # ray stop --force
    assert removed == ["/tmp/ray"]                                  # cached GCS session cleared
