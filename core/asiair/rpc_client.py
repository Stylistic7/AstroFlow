"""Tiny synchronous JSON-RPC client for the ASIAir's proprietary
control protocol.

The ASIAir exposes a line-delimited JSON-RPC 2.0 server on:
  * TCP 4400 — "ASI AIR guider"
  * TCP 4700 — "ASI AIR imager"

Both ports emit a `{"Event":"Version",...}` banner on connect (no
request needed). Requests are `{"id": N, "method": "...", "params":
[...]}\r\n`; responses are `{"jsonrpc": "2.0", "method": "...",
"result": ..., "code": 0, "id": N}` on success, or carry an `"error"`
+ non-zero `"code"` on failure.

Reverse-engineered against firmware 13.41 via direct probing of a
live device; see the planning notes in
`~/.claude/plans/splendid-munching-rainbow.md` for the discovered
method vocabulary.

Single concern: open a TCP connection, send one request, parse one
response, return it (or None on any failure). The live data is a
best-effort overlay — never blocks or raises into callers.
"""
from __future__ import annotations

import json
import socket
import structlog
from typing import Any


log = structlog.get_logger()


# Any Event push, not just the connect-time Version banner: firmware
# 14.39 interleaves unsolicited PiStatus (and likely other) events on
# the imager port, and treating one as the reply poisoned snapshots.
_BANNER_PREFIX = b'{"Event":'
_DEFAULT_TIMEOUT_S = 2.0
_READ_CHUNK = 16384


def _read_one_json_line(sock: socket.socket) -> bytes | None:
    """Read bytes until we see a complete top-level JSON object
    terminated by `\\n`. Tolerates the banner re-emit the imager port
    occasionally interleaves between responses by skipping any line
    that starts with `_BANNER_PREFIX` (an `Event` push, not a reply
    to OUR request) and reading the next one.
    """
    buf = b""
    while True:
        chunk = sock.recv(_READ_CHUNK)
        if not chunk:
            return buf or None
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith(_BANNER_PREFIX):
                # Banner / Event push; not the reply to OUR request.
                continue
            return line


def call(
    ip: str,
    port: int,
    method: str,
    params: list | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> dict | None:
    """Send one JSON-RPC method call to (ip, port) and return the
    parsed response dict. Returns `None` for ANY failure (connection
    refused, timeout, malformed reply, error code in the response).

    Caller-side rule: never raise. The panel must keep rendering its
    offline-derived fields when the device is unreachable.
    """
    if params is None:
        params = []
    request = (
        json.dumps({"id": 1, "method": method, "params": params}, separators=(",", ":"))
        + "\r\n"
    ).encode()

    sock: socket.socket | None = None
    try:
        sock = socket.create_connection((ip, port), timeout=timeout)
        sock.settimeout(timeout)
        # Drain the connect-time Version banner so the response we
        # read next is for OUR request. The banner ends in `\n`; one
        # recv usually catches it. If timing makes it land in the
        # middle of the read, `_read_one_json_line` will skip it.
        try:
            sock.recv(_READ_CHUNK)
        except socket.timeout:
            return None

        sock.sendall(request)
        line = _read_one_json_line(sock)
        if line is None:
            return None
        reply = json.loads(line.decode("utf-8", errors="replace"))
        if not isinstance(reply, dict):
            return None
        if reply.get("code", 0) != 0 or "error" in reply:
            log.debug(
                "asiair rpc error",
                ip=ip, port=port, method=method,
                code=reply.get("code"), error=reply.get("error"),
            )
            return None
        return reply
    except (OSError, socket.timeout, json.JSONDecodeError, ValueError) as e:
        log.debug(
            "asiair rpc failed",
            ip=ip, port=port, method=method, error=str(e),
        )
        return None
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def call_many(
    ip: str,
    port: int,
    methods: list[tuple[str, list]],
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> dict[str, dict | None]:
    """Convenience: send several calls and return `{method: result}`.

    Each call opens its own connection — the device DOES accept
    multiple requests per connection, but the banner re-emit and the
    occasional error-induced close make a per-call connection simpler
    and only ~5-10 ms slower per call on a LAN, which is fine at our
    polling cadence.
    """
    return {method: call(ip, port, method, params, timeout)
            for method, params in methods}
