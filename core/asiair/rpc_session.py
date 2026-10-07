"""A persistent, optionally authenticated JSON-RPC session with one ASIAir port.

`rpc_client.call` opens a socket per request, which is right for the
poller's six unauthenticated reads a minute. The handshake is
PER CONNECTION, so anything gated — plans, sequences, the wheel — needs
a socket that stays open after `verify_client`, and the device's
unsolicited Events (sequence progress, dithers, PiStatus) only reach a
listener that keeps reading. This is that socket.

Wire format (see `rpc_client`): newline-terminated JSON; requests carry
an `id`, replies echo it, Events have none. One reader thread splits the
stream: replies go to the waiting caller by id, everything else to an
event queue the owner drains at leisure.

Safety: nothing here moves anything. `call` sends whatever it is given;
the caller decides. The CLI in `scripts/asiair_rpc.py` refuses `set_*` /
`start_*` / `import_*` methods unless told otherwise.
"""
from __future__ import annotations

import json
import queue
import socket
import threading
from pathlib import Path
from typing import Any

import structlog

from asiair.rsa_sign import RsaPrivateKey, load_key_file, sign_challenge

log = structlog.get_logger()

IMAGER_PORT = 4700
GUIDER_PORT = 4400
_READ_CHUNK = 65536
_DEFAULT_TIMEOUT_S = 5.0
_MAX_EVENTS = 10_000


class AsiairRpcError(RuntimeError):
    """The device answered with a non-zero `code` / an `error`."""

    def __init__(self, method: str, reply: dict):
        self.method, self.reply = method, reply
        super().__init__(f"{method}: code {reply.get('code')} {reply.get('error') or ''}".rstrip())


class AsiairSession:
    def __init__(self, ip: str, port: int = IMAGER_PORT, *, timeout: float = _DEFAULT_TIMEOUT_S):
        self.ip, self.port, self.timeout = ip, port, timeout
        self.verified = False
        self._sock: socket.socket | None = None
        self._next_id = 0
        self._pending: dict[int, tuple[threading.Event, list]] = {}
        self._lock = threading.Lock()
        self._events: queue.Queue = queue.Queue(maxsize=_MAX_EVENTS)
        self._alive = False
        self._reader: threading.Thread | None = None

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> "AsiairSession":
        self._sock = socket.create_connection((self.ip, self.port), timeout=self.timeout)
        self._sock.settimeout(None)
        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, name=f"asiair-{self.port}", daemon=True)
        self._reader.start()
        return self

    def close(self) -> None:
        self._alive = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=1.0)

    def __enter__(self) -> "AsiairSession":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def alive(self) -> bool:
        return self._alive

    # ------------------------------------------------------------ reading
    def _read_loop(self) -> None:
        buf = b""
        try:
            while self._alive and self._sock is not None:
                chunk = self._sock.recv(_READ_CHUNK)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if line:
                        self._dispatch(line)
        except OSError:
            pass
        finally:
            self._alive = False
            with self._lock:
                for ev, slot in self._pending.values():
                    slot.append(None)
                    ev.set()

    def _dispatch(self, line: bytes) -> None:
        try:
            msg = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            log.debug("asiair: unparseable line", port=self.port, line=line[:200])
            return
        if isinstance(msg, dict) and "id" in msg:
            with self._lock:
                waiter = self._pending.get(msg["id"])
            if waiter is not None:
                waiter[1].append(msg)
                waiter[0].set()
                return
        try:
            self._events.put_nowait(msg)
        except queue.Full:
            pass

    # ------------------------------------------------------------ calling
    def call(self, method: str, params: list | None = None, *, timeout: float | None = None) -> Any:
        """Send one request and return its `result`. Raises `AsiairRpcError`
        on a device error, `TimeoutError` when nothing comes back (the gate
        drops unauthenticated calls in silence — no reply IS the signal)."""
        if self._sock is None or not self._alive:
            raise ConnectionError(f"session to {self.ip}:{self.port} is not open")
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            waiter: tuple[threading.Event, list] = (threading.Event(), [])
            self._pending[rid] = waiter
        request = {"id": rid, "method": method, "params": params or []}
        try:
            self._sock.sendall((json.dumps(request, separators=(",", ":")) + "\r\n").encode())
            if not waiter[0].wait(timeout or self.timeout):
                raise TimeoutError(f"{method}: no reply from {self.ip}:{self.port} within {timeout or self.timeout}s")
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        reply = waiter[1][0]
        if reply is None:
            raise ConnectionError(f"{method}: connection closed")
        if reply.get("code", 0) != 0 or "error" in reply:
            raise AsiairRpcError(method, reply)
        return reply.get("result")

    def events(self) -> list[Any]:
        """Drain every Event received so far (oldest first)."""
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # ------------------------------------------------------------ auth
    def verify(self, key: RsaPrivateKey | Path | None = None) -> bool:
        """Run the challenge handshake on THIS connection. `key` is a parsed
        key, a PEM path, or None for the default key file."""
        if not isinstance(key, RsaPrivateKey):
            key = load_key_file(key)
        challenge = (self.call("get_verify_str") or {}).get("str")
        if not challenge:
            raise AsiairRpcError("get_verify_str", {"code": -1, "error": "no challenge"})
        # Two positional strings — the device answers `105 expected string
        # param` to a lone signature (firmware 14.39).
        self.call("verify_client", [sign_challenge(key, challenge), challenge])
        self.verified = bool(self.call("pi_is_verified"))
        log.info("asiair handshake", ip=self.ip, port=self.port, verified=self.verified)
        return self.verified
