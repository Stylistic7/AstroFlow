"""mount_head_shares: zombie-mount eviction (post-reboot join regression).

`os.path.ismount` only consults the mount table; after a head reboot a
worker can hold a table entry whose SMB session is DEAD — reads hang or
error. The join must detect dead, force-unmount, and remount at the
canonical path, or the worker joins Ray with unreadable disks and every
task fails on missing inputs (2026-08-10)."""
import time
from pathlib import Path

import state.worker_sync as ws


def test_zombie_mount_is_evicted_and_remounted(monkeypatch):
    calls = []
    monkeypatch.setattr(ws, "_is_mounted", lambda mp: True)
    alive = iter([False])  # dead on first probe; default True afterwards
    monkeypatch.setattr(ws, "_mount_is_alive",
                        lambda mp, timeout_s=5.0: next(alive, True))
    monkeypatch.setattr(ws, "_force_unmount",
                        lambda mp: calls.append(("unmount", str(mp))))

    def fake_run(cmd, **kw):
        calls.append(("mount", cmd[-1]))
        class R:
            returncode = 0
            stderr = ""
        return R()
    monkeypatch.setattr(ws.subprocess, "run", fake_run)

    ws.mount_head_shares("192.168.50.2", ["Astro"])
    kinds = [k for k, _ in calls]
    assert "unmount" in kinds, "dead mount must be evicted"
    assert "mount" in kinds, "must remount after eviction"


def test_live_mount_is_left_alone(monkeypatch):
    calls = []
    monkeypatch.setattr(ws, "_is_mounted", lambda mp: True)
    monkeypatch.setattr(ws, "_mount_is_alive", lambda mp, timeout_s=5.0: True)
    monkeypatch.setattr(ws, "_force_unmount",
                        lambda mp: calls.append("unmount"))
    monkeypatch.setattr(ws.subprocess, "run",
                        lambda *a, **k: calls.append("mount"))
    ws.mount_head_shares("192.168.50.2", ["Astro"])
    assert calls == [], "an alive mount must not be touched"


def test_dead_probe_times_out_instead_of_hanging(monkeypatch):
    def hanging_listdir(mp):
        time.sleep(30)
    monkeypatch.setattr(ws.os, "listdir", hanging_listdir)
    t0 = time.monotonic()
    assert ws._mount_is_alive(Path("/tmp"), timeout_s=0.5) is False
    assert time.monotonic() - t0 < 3, "probe must time out, not hang"
