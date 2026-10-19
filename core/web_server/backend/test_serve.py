"""`web_server.backend.serve`: the Windows web server process, run here under
the same waitress. The served app and the start hook are this module's."""
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import flask

_CORE = Path(__file__).resolve().parents[2]
STARTED_FILE_ENV = "ASTROFLOW_TEST_SERVE_STARTED"

hello_app = flask.Flask("test_serve")


@hello_app.route("/")
def hello():
    return "hello from waitress"


def mark_started():
    Path(os.environ[STARTED_FILE_ENV]).write_text("started")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_runs_the_named_app_after_the_start_hook():
    port = _free_port()
    started = Path(tempfile.mkdtemp()) / "started"
    server = subprocess.Popen(
        [sys.executable, "-m", "web_server.backend.serve", "web_server.backend.test_serve:hello_app",
         "--port", str(port), "--threads", "2", "--on-start", "web_server.backend.test_serve:mark_started"],
        cwd=_CORE, env={**os.environ, STARTED_FILE_ENV: str(started)})
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).read()
                break
            except OSError:
                assert server.poll() is None, "the server exited"
                assert time.monotonic() < deadline, "the server never answered"
                time.sleep(0.2)
        assert body == b"hello from waitress"
        assert started.read_text() == "started", "the hook ran in the server's process"
    finally:
        server.terminate()
        server.wait(timeout=10)
