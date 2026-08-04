"""Worker pull-deploy logic: code-sync gating + mount/unmount command shape.
Ray-free and mount-free — subprocess and HTTP are stubbed.
"""
import subprocess

import pytest

import state.worker_sync as ws


def test_head_current_version_reads_cluster_version_endpoint(monkeypatch):
    # The version comes from the head's AUTHORITATIVE HTTP endpoint — never a
    # zeroconf TXT (which a killed prior head leaves as a ghost with a stale SHA).
    import json as _json
    payload = _json.dumps({"cluster_version": "c" * 40}).encode()

    class _R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return payload

    seen = {}

    def fake_urlopen(url, timeout=0):
        seen["url"] = url
        return _R()

    monkeypatch.setattr(ws.urllib.request, "urlopen", fake_urlopen)
    assert ws.head_current_version("10.0.0.53") == "c" * 40
    assert seen["url"] == f"http://10.0.0.53:{ws.HTTP_PORT}/api/cluster-version"


def test_sync_to_head_skips_non_deployable(monkeypatch):
    # "dev"/"dirty"/"" head -> None, and NEVER pulls a snapshot.
    monkeypatch.setattr(ws, "pull_head_snapshot",
                        lambda ip: pytest.fail("must not pull for a non-deployable head"))
    for v in ("dev", "dirty", "", "nothex" * 7):
        monkeypatch.setattr(ws, "head_current_version", lambda ip, _v=v: _v)
        assert ws.sync_to_head("10.0.0.53") is None


def test_sync_to_head_noop_when_already_at_head_sha(monkeypatch):
    sha = "a" * 40
    monkeypatch.setattr(ws, "head_current_version", lambda ip: sha)
    monkeypatch.setattr(ws, "current_synced_sha", lambda: sha)
    monkeypatch.setattr(ws, "pull_head_snapshot", lambda ip: pytest.fail("should not pull"))
    assert ws.sync_to_head("10.0.0.53") == sha


def test_sync_to_head_pulls_current_when_behind(monkeypatch):
    monkeypatch.setattr(ws, "head_current_version", lambda ip: "b" * 40)
    monkeypatch.setattr(ws, "current_synced_sha", lambda: "a" * 40)
    monkeypatch.setattr(ws, "pull_head_snapshot", lambda ip: "b" * 40)
    assert ws.sync_to_head("10.0.0.53") == "b" * 40


def test_pull_head_snapshot_uses_bare_url_and_trusts_header(monkeypatch, tmp_path):
    class _Resp:
        headers = {"X-AstroFlow-SHA": "b" * 40}
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

    assert ws.pull_head_snapshot("10.0.0.53") == "b" * 40
    # NO ?sha= — asks for CURRENT and records the SHA the head STAMPED in the
    # header, so a stale/ghost advertised SHA can never make it request the wrong one.
    assert seen["url"] == f"http://10.0.0.53:{ws.HTTP_PORT}/api/code-snapshot"
    assert extracted["tar"] == b"TARBYTES" and extracted["sha"] == "b" * 40


def test_mount_uses_explicit_ip_no_user_and_volumes_path(monkeypatch):
    cmds = []
    mounted = {"Astro": False}

    def fake_run(cmd, **kw):
        cmds.append(cmd)
        mounted["Astro"] = True  # the mount command "succeeds"
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ws.subprocess, "run", fake_run)
    monkeypatch.setattr(ws.os.path, "ismount",
                        lambda p: mounted.get(str(p).rsplit("/", 1)[-1], False))

    ws.mount_head_shares("10.0.0.53", ["Astro"])

    assert len(cmds) == 1
    assert cmds[0][0] == "osascript"
    # Explicit IP (forces the 10 GbE link), and NO user in the URL — an explicit
    # user@/guest@ made the mount silently fail to appear; bare IP uses the
    # current login session (same Apple ID).
    assert 'mount volume "smb://10.0.0.53/Astro"' in cmds[0][-1]
    assert "@" not in cmds[0][-1]


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
