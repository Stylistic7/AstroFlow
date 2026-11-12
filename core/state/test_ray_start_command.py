"""`ray start` runs as an argument list with its own environment
(`state.ray_control._start_ray`): no shell, which Windows cannot run the old
`ulimit … && VAR=x ray start … --resources '<json>'` string in."""
import json
import sys

import state.ray_control as rc


def test_the_head_starts_ray_without_a_shell(monkeypatch):
    seen = {}

    class _Popen:
        def __init__(self, command, **kwargs):
            seen["command"], seen["kwargs"] = command, kwargs

    monkeypatch.setattr(rc.subprocess, "Popen", _Popen)
    monkeypatch.setattr(rc, "_wait_for_ray", lambda process, timeout: None)
    monkeypatch.setattr(rc, "_ray_memory_bytes", lambda: 123)
    monkeypatch.setattr(rc, "_workers_soft_limit", lambda: 7)
    monkeypatch.setattr(rc.os.path, "exists", lambda p: False)       # no Siril
    monkeypatch.setattr(rc, "_fresh_temp_dir", lambda: None)
    rc.start_ray_head_node(node_ip_address="192.168.50.2")
    command, kwargs = seen["command"], seen["kwargs"]
    assert isinstance(command, list) and "shell" not in kwargs
    assert command[len(rc.RAY_CLI):][:4] == ["start", "--head", "--node-ip-address", "192.168.50.2"]
    resources = json.loads(command[command.index("--resources") + 1])     # plain JSON, no shell quotes
    assert resources["diskcache"] == 1 and "Siril" not in resources
    assert command[command.index("--memory") + 1] == "123"
    assert kwargs["env"]["RAY_num_workers_soft_limit"] == "7"
    assert kwargs["env"]["RAY_TMPDIR"] == rc.ray_temp_root()


def test_ray_keeps_its_session_outside_tmp():
    """macOS's tmp_cleaner empties /tmp of files untouched for three days; on
    2026-11-12 that was the shipped core/ of a head four days up. The sockets
    under the session must still fit an AF_UNIX path (104 bytes on macOS)."""
    root = rc.ray_temp_root()
    assert not root.startswith("/tmp/")
    if sys.platform != "win32":
        assert len(f"{root}/ray/session_{'x' * 32}/sockets/plasma_store") <= 103


def test_a_start_keeps_only_the_newest_session(monkeypatch, tmp_path):
    root = tmp_path / "ray"
    old, new = root / "session_2026-11-07_13-06-56_042165_61384", root / "session_2026-11-10_08-00-00_000000_1"
    old.mkdir(parents=True), new.mkdir()
    (old / "runtime_resources").mkdir()
    (root / "session_latest").symlink_to(new)
    (root / "ray_current_cluster").write_text("192.168.50.2:6379")
    monkeypatch.setattr(rc, "ray_temp_root", lambda: str(tmp_path))
    monkeypatch.setattr(rc, "is_ray_connected", lambda: False)
    rc._fresh_temp_dir()
    assert sorted(p.name for p in root.iterdir()) == [
        "ray_current_cluster", "session_2026-11-10_08-00-00_000000_1", "session_latest"]


def test_a_live_local_node_keeps_every_session(monkeypatch, tmp_path):
    root = tmp_path / "ray"
    for name in ("session_2026-11-07_13-06-56_042165_61384", "session_2026-11-10_08-00-00_000000_1"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(rc, "ray_temp_root", lambda: str(tmp_path))
    monkeypatch.setattr(rc, "is_ray_connected", lambda: True)
    rc._fresh_temp_dir()
    assert len(list(root.iterdir())) == 2


def test_a_ray_start_that_exits_with_an_error_fails_at_once(monkeypatch):
    import pytest

    class _Exited:
        returncode = 2

        def poll(self):
            return 2

    monkeypatch.setattr(rc, "is_ray_connected", lambda: False)
    with pytest.raises(RuntimeError, match="exited with 2"):
        rc._wait_for_ray(_Exited(), timeout=60)
