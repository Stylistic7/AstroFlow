"""The MCP sidecar, supervised like a gunicorn server."""
import os
import subprocess
import sys

from mcp_server import MCP_PORT
from state.gunicorn_control import GunicornConfig, GunicornManager
from state.procs import CHILD_FLAGS


class McpManager(GunicornManager):
    """One `python -m mcp_server` process on `MCP_PORT`, with gunicorn's
    boot-heal rounds and stop — and the same short cache-op timeout, since
    it reads the head's diskcache exactly as a web worker does."""

    def __init__(self, port: int = MCP_PORT):
        super().__init__(GunicornConfig(
            name="mcp_server", config_file="", module="mcp_server",
            worker_type="", num_servers=1, start_port=port, num_workers=0,
            timeout=0,
        ))

    def _spawn(self, port: int) -> subprocess.Popen:
        command = [sys.executable, "-m", "mcp_server", "--port", str(port)]
        print(f"### Starting MCP sidecar with command: {command}")
        env = os.environ | {"ASTROFLOW_CACHE_TIMEOUT_S": "15"}
        return subprocess.Popen(command, env=env, creationflags=CHILD_FLAGS)
