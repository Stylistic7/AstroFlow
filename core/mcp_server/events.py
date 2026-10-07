"""The head's event stream, as a polling tool.

A daemon thread is an ordinary SSE client of the stream server
(http://127.0.0.1:8001/stream) — the same feed the browser gets — and keeps
the last events in a ring buffer. It never touches the announcer's diskcache
queue, which has exactly one consumer (the stream server itself). Heartbeats
are noted, not buffered. Chosen over MCP notifications: the transport is
stateless HTTP and the client polls.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator

from mcp_server.server import tool

DEFAULT_STREAM_URL = "http://127.0.0.1:8001/stream"
MAX_WAIT_S = 60.0

_buffer: deque = deque(maxlen=1000)
_cv = threading.Condition()
_state = {"connected": False, "last_heartbeat_at": None, "last_error": None,
          "last_seen": {}}


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, str | None]]:
    """(event, data) pairs from SSE lines (`stream.format_sse` framing:
    `event: x`, optional `data: y`, blank line ends the message)."""
    event, data = None, None
    for line in lines:
        if line == "":
            if event is not None:
                yield event, data
            event, data = None, None
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    if event is not None:
        yield event, data


def _record(event: str, data: str | None) -> None:
    now = time.time()
    with _cv:
        _state["last_seen"][event] = now
        if event == "heartbeat":
            _state["last_heartbeat_at"] = now
        else:
            _buffer.append({"at": now, "event": event, "data": data})
        _cv.notify_all()


def _listen(url: str) -> None:
    import requests
    backoff = 1.0
    while True:
        try:
            with requests.get(url, stream=True, timeout=(5, 90)) as r:
                r.raise_for_status()
                with _cv:
                    _state["connected"] = True
                    _state["last_error"] = None
                backoff = 1.0
                for event, data in parse_sse(r.iter_lines(decode_unicode=True)):
                    _record(event, data)
        except Exception as e:  # reconnect forever; the tool reports the error
            with _cv:
                _state["connected"] = False
                _state["last_error"] = f"{type(e).__name__}: {e}"
        time.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


def start_listener(url: str = DEFAULT_STREAM_URL) -> threading.Thread:
    t = threading.Thread(target=_listen, args=(url,), name="sse-listener", daemon=True)
    t.start()
    return t


def last_seen(event: str) -> float | None:
    with _cv:
        return _state["last_seen"].get(event)


@tool()
def events(last_n: int = 50, event: str | None = None,
           wait_for: str | None = None, timeout_s: float = 30.0) -> dict:
    """Recent events from the head (ray_status, compute_graph_status every
    5 s, reload_data after a re-plan, asiair_devices), newest last, with
    unix timestamps. `event` filters by name. `wait_for=<name>` blocks up
    to `timeout_s` (<= 60) for the next such event — e.g. wait_for=
    "reload_data" after a confirmed write. Also reports the listener's
    connection state and last heartbeat.
    """
    deadline = time.time() + max(0.0, min(timeout_s, MAX_WAIT_S))
    with _cv:
        if wait_for is not None:
            since = _state["last_seen"].get(wait_for)
            while _state["last_seen"].get(wait_for) == since:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                _cv.wait(remaining)
        rows = [e for e in _buffer if event is None or e["event"] == event]
        return {
            "connected": _state["connected"],
            "last_error": _state["last_error"],
            "last_heartbeat_at": _state["last_heartbeat_at"],
            "waited_for": wait_for,
            "arrived": (wait_for is not None
                        and _state["last_seen"].get(wait_for) != since),
            "events": rows[-max(1, last_n):],
        }
