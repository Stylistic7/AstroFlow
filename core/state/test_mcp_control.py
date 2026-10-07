"""The MCP sidecar's lifecycle: spawned like a gunicorn server on the head
only, healed at boot, stopped at quit, killed by the belt-and-braces."""
import os
import subprocess
import sys

from state.mcp_control import MCP_PORT, McpManager


class _Alive:
    pid = 4242
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self):
        pass


def test_spawn_runs_the_sidecar_with_the_web_workers_cache_timeout(monkeypatch):
    seen = {}

    def fake_popen(command, env=None, **kw):
        seen["command"], seen["env"] = command, env
        return _Alive()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    McpManager()._spawn(MCP_PORT)
    assert seen["command"] == [sys.executable, "-m", "mcp_server", "--port", str(MCP_PORT)]
    assert seen["env"]["ASTROFLOW_CACHE_TIMEOUT_S"] == "15"
    assert seen["env"].keys() >= os.environ.keys()


def test_start_spawns_one_healthy_process(monkeypatch):
    import state.gunicorn_control as gc
    monkeypatch.setattr(gc, "_BOOT_SETTLE_S", 0.0)
    spawned = []
    manager = McpManager()
    monkeypatch.setattr(manager, "_spawn", lambda port: spawned.append(port) or _Alive())
    manager.start()
    assert spawned == [MCP_PORT] and manager.ports == [MCP_PORT]


def test_head_starts_and_quits_the_sidecar(monkeypatch):
    import state.state_machine as sm
    monkeypatch.setattr(sm, "is_head_node", lambda: True)
    _, actions = sm._init({}, {"type": "init"})
    types = [a["type"] for a in actions]
    assert types.index("start_gunicorn") < types.index("start_mcp")
    monkeypatch.setattr(sm, "is_head_node", lambda: False)
    _, actions = sm._init({}, {"type": "init"})
    assert "start_mcp" not in [a["type"] for a in actions]
    _, quit_actions = sm._quit({}, {"type": "quit"})
    assert "stop_mcp" in [a["type"] for a in quit_actions]
    assert {"start_mcp", "stop_mcp"} <= sm._HANDLERS.keys()


def test_stop_mcp_tolerates_a_missing_manager():
    import state.state_machine as sm
    state, actions = sm._stop_mcp({}, {"type": "stop_mcp"})
    assert state == {} and actions == []
