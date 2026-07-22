"""Unit tests for the Ray-control helpers: the bounded `is_ray_connected`, a
local start/stop lifecycle barrier — `ray status` capped with a timeout so a
dead remote head can't hang `_wait_for_ray` / `stop_ray_node`. We do NOT probe a
running connection's health on top of Ray; that flaky check tore down healthy
raylets, so it's gone.
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
