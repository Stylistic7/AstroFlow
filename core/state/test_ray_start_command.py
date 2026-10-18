"""`ray start` runs as an argument list with its own environment
(`state.ray_control._start_ray`): no shell, which Windows cannot run the old
`ulimit … && VAR=x ray start … --resources '<json>'` string in."""
import json

import state.ray_control as rc


def test_the_head_starts_ray_without_a_shell(monkeypatch):
    seen = {}

    class _Popen:
        def __init__(self, command, **kwargs):
            seen["command"], seen["kwargs"] = command, kwargs

    monkeypatch.setattr(rc.subprocess, "Popen", _Popen)
    monkeypatch.setattr(rc, "_wait_for_ray", lambda timeout: None)
    monkeypatch.setattr(rc, "_ray_memory_bytes", lambda: 123)
    monkeypatch.setattr(rc, "_workers_soft_limit", lambda: 7)
    monkeypatch.setattr(rc.os.path, "exists", lambda p: False)       # no Siril
    rc.start_ray_head_node(node_ip_address="192.168.50.2")
    command, kwargs = seen["command"], seen["kwargs"]
    assert isinstance(command, list) and "shell" not in kwargs
    assert command[len(rc.RAY_CLI):][:4] == ["start", "--head", "--node-ip-address", "192.168.50.2"]
    resources = json.loads(command[command.index("--resources") + 1])     # plain JSON, no shell quotes
    assert resources["diskcache"] == 1 and "Siril" not in resources
    assert command[command.index("--memory") + 1] == "123"
    assert kwargs["env"]["RAY_num_workers_soft_limit"] == "7"
