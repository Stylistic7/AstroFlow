"""The app servers must not be able to wedge on a dead client's socket.

gthread heartbeats from its MAIN thread, so `--timeout 300` reaps nothing
while all four pool threads sit blocked in `sendall` to browsers that
navigated away — the recurring "web server is wedged again" of 2026-09-08/09,
clearable only by a restart. A per-socket-operation timeout makes gunicorn's
existing error path reachable.
"""
import socket

import pytest

from state.gunicorn_configs import guincorn_configs
from web_server.backend import gunicorn_app_config as cfg


@pytest.fixture
def unpatched(monkeypatch):
    """`post_fork` mutates `TConn` process-wide; put it back afterwards."""
    from gunicorn.workers.gthread import TConn
    monkeypatch.setattr(TConn, "init", TConn.init)
    monkeypatch.setattr(cfg, "_patched", False)
    return TConn


def _connected_pair():
    a, b = socket.socketpair()
    return a, b


def test_post_fork_bounds_socket_operations(unpatched):
    TConn = unpatched
    cfg.post_fork(None, None)

    a, b = _connected_pair()
    try:
        conn = TConn.__new__(TConn)
        conn.sock = a
        conn.initialized = True          # the real body is gunicorn's business
        conn.init()
        assert a.gettimeout() == cfg.SOCKET_TIMEOUT_S
    finally:
        a.close(); b.close()


def test_the_timeout_survives_the_blocking_mode_handle_forces(unpatched):
    """`ThreadWorker.handle` calls `setblocking(True)` — which CLEARS any
    timeout — immediately before `conn.init()`. That ordering is the whole
    reason the patch lives on `init` and not on the listener or the accept."""
    TConn = unpatched
    cfg.post_fork(None, None)

    a, b = _connected_pair()
    try:
        conn = TConn.__new__(TConn)
        conn.sock = a
        conn.initialized = True
        a.setblocking(True)              # what handle() does
        assert a.gettimeout() is None
        conn.init()
        assert a.gettimeout() == cfg.SOCKET_TIMEOUT_S
    finally:
        a.close(); b.close()


def test_post_fork_is_idempotent(unpatched):
    TConn = unpatched
    cfg.post_fork(None, None)
    once = TConn.init
    cfg.post_fork(None, None)
    assert TConn.init is once, "post_fork stacked a second wrapper"


def test_the_app_servers_load_the_hook():
    """A gunicorn config file only applies where it is passed with -c."""
    configs = guincorn_configs()
    for name in ("static_server", "tile_server"):
        assert configs[name].config_file == \
            "web_server/backend/gunicorn_app_config.py"
    # the gevent stream server keeps its own (announcer) post_fork
    assert configs["stream_server"].config_file == \
        "web_server/backend/gunicorn_config.py"


def test_the_socket_timeout_sits_under_the_arbiter_timeout():
    """The socket must give up FIRST — otherwise the arbiter kills the whole
    worker (and every other request it is serving) instead of one connection."""
    configs = guincorn_configs()
    for name in ("static_server", "tile_server"):
        assert cfg.SOCKET_TIMEOUT_S < configs[name].timeout
