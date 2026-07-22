"""Worker pull-deploy logic: code-sync gating + mount/unmount command shape.
Ray-free and mount-free — subprocess and HTTP are stubbed.
"""
import subprocess

import pytest

import state.worker_sync as ws


def test_sync_refuses_non_deployable_versions(monkeypatch):
    # "dev"/"dirty"/"" must never trigger a download — they keep the cluster
    # single-machine.
    called = {"n": 0}
    monkeypatch.setattr(ws.urllib.request, "urlopen",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    for v in ("dev", "dirty", "", "nothex" * 7):
        assert ws.sync_to_version("10.0.0.53", v) is False
    assert called["n"] == 0


def test_sync_is_noop_when_already_at_sha(monkeypatch):
    sha = "a" * 40
    monkeypatch.setattr(ws, "current_synced_sha", lambda: sha)
    monkeypatch.setattr(ws.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("should not download"))
    assert ws.sync_to_version("10.0.0.53", sha) is True


def test_sync_downloads_and_extracts_when_behind(monkeypatch, tmp_path):
    sha = "b" * 40
    monkeypatch.setattr(ws, "current_synced_sha", lambda: "a" * 40)

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"TARBYTES"

    seen = {}

    def fake_urlopen(url, timeout=0):
        seen["url"] = url
        return _Resp()

    monkeypatch.setattr(ws.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ws, "checkout_dir", lambda: tmp_path / "checkout")
    extracted = {}
    monkeypatch.setattr(ws, "verify_and_extract",
                        lambda tar, s, dest: extracted.update(tar=tar, sha=s, dest=dest))

    assert ws.sync_to_version("10.0.0.53", sha) is True
    # Asks the head for that exact SHA (race-proof) over its 10 GbE IP + port.
    assert seen["url"] == f"http://10.0.0.53:{ws.HTTP_PORT}/api/code-snapshot?sha={sha}"
    assert extracted["tar"] == b"TARBYTES" and extracted["sha"] == sha


def test_mount_uses_explicit_ip_current_user_and_volumes_path(monkeypatch):
    cmds = []
    mounted = {"Astro": False}

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        mounted["Astro"] = True  # the mount command "succeeds"
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    monkeypatch.setattr(ws.getpass, "getuser", lambda: "nobody")
    monkeypatch.setattr(ws.os.path, "ismount",
                        lambda p: mounted.get(str(p).rsplit("/", 1)[-1], False))

    ws.mount_head_shares("10.0.0.53", ["Astro"])

    assert len(cmds) == 1
    assert cmds[0][0] == "osascript"
    # Mounts as the current user (same account across the LAN), never `guest`.
    assert 'mount volume "smb://nobody@10.0.0.53/Astro"' in cmds[0][-1]
    assert "guest@" not in cmds[0][-1]


def test_mount_is_skipped_when_already_mounted(monkeypatch):
    monkeypatch.setattr(ws.os.path, "ismount", lambda p: True)
    monkeypatch.setattr(ws.subprocess, "run",
                        lambda *a, **k: pytest.fail("should not mount"))
    ws.mount_head_shares("10.0.0.53", ["Astro", "Astro2"])  # no-op


def test_mount_raises_on_osascript_failure(monkeypatch):
    monkeypatch.setattr(ws.os.path, "ismount", lambda p: False)
    monkeypatch.setattr(
        ws.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "auth error"),
    )
    with pytest.raises(RuntimeError, match="failed"):
        ws.mount_head_shares("10.0.0.53", ["Astro"])


def test_unmount_only_touches_mounted_shares(monkeypatch):
    cmds = []
    monkeypatch.setattr(ws.os.path, "ismount",
                        lambda p: str(p).endswith("Astro"))  # Astro mounted, Astro2 not
    monkeypatch.setattr(ws.subprocess, "run",
                        lambda cmd, **kw: cmds.append(cmd))
    ws.unmount_head_shares(["Astro", "Astro2"])
    assert len(cmds) == 1
    assert cmds[0][:2] == ["diskutil", "unmount"]
    assert cmds[0][-1].endswith("/Volumes/Astro")
