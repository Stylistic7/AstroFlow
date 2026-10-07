"""`AsiairSession` against a fake ASIAir: banner on connect, Events
interleaved with replies, id matching, the silent gate, and the handshake.
`rsa_sign` against openssl: PKCS#1 v1.5 / SHA-1 is deterministic, so our
signature must equal `openssl dgst -sha1 -sign` byte for byte.

Plain asserts and a hand-rolled server so the file also runs as a script
(`PYTHONPATH=. python asiair/test_rpc_session.py` from `core/`) — pytest
resets the shared Ray and must wait for a restart window; the protocol
work can't.
"""
from __future__ import annotations

import json
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path

from asiair.rpc_session import AsiairRpcError, AsiairSession
from asiair.rsa_sign import load_private_key, sign_challenge, sign_sha1_pkcs1v15

BANNER = b'{"Event":"Version","Timestamp":"1.0","name":"ASI AIR imager","firmware_ver_string":"14.39"}\n'


class FakeAir:
    """Answers like firmware 14.39: `test_connection` and `get_verify_str`
    always; everything else only after `verify_client`, and before that —
    silence, never an error."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.seen: list[dict] = []
        self.verified = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.srv.accept()
        conn.sendall(BANNER)
        buf = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                req = json.loads(line)
                self.seen.append(req)
                for out in self._answer(req):
                    conn.sendall(json.dumps(out).encode() + b"\n")

    def _answer(self, req):
        m, rid = req["method"], req["id"]
        ok = lambda result: {"jsonrpc": "2.0", "method": m, "result": result, "code": 0, "id": rid}
        if m == "test_connection":
            # an Event lands between request and reply, as the real device does
            yield {"Event": "PiStatus", "Timestamp": "2.0", "temp": 41.5}
            yield ok("server connected!")
        elif m == "get_verify_str":
            yield ok({"str": "challenge-xyz"})
        elif m == "verify_client":
            p = req["params"]
            if len(p) != 2 or not all(isinstance(x, str) for x in p):
                yield {"jsonrpc": "2.0", "method": m, "error": "expected string param", "code": 105, "id": rid}
                return
            self.verified = len(p[0]) > 40 and p[1] == "challenge-xyz"
            yield ok(0)
        elif m == "pi_is_verified":
            yield ok(self.verified)
        elif not self.verified:
            return  # the gate: dropped in silence
        elif m == "fails":
            yield {"jsonrpc": "2.0", "method": m, "error": "boom", "code": 109, "id": rid}
        else:
            yield ok({"echo": req["params"]})


def _test_key_pem() -> bytes:
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["openssl", "genrsa", "-traditional", "-out", str(tmp / "k.pem"), "2048"], check=True, capture_output=True)
    return (tmp / "k.pem").read_bytes()


def test_signature_matches_openssl():
    pem = _test_key_pem()
    key = load_private_key(pem)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "k.pem").write_bytes(pem)
    msg = b"ECcwuG3cYCRLJKswSN5u4-Lsm-7Hj9Jv"
    (tmp / "msg").write_bytes(msg)
    ref = subprocess.run(
        ["openssl", "dgst", "-sha1", "-sign", str(tmp / "k.pem"), str(tmp / "msg")],
        check=True, capture_output=True,
    ).stdout
    assert sign_sha1_pkcs1v15(key, msg) == ref
    assert len(sign_challenge(key, msg.decode())) > 300  # base64 of 256 bytes


def test_pkcs8_pem_also_loads():
    pem = _test_key_pem()
    tmp = Path(tempfile.mkdtemp())
    (tmp / "k.pem").write_bytes(pem)
    p8 = subprocess.run(["openssl", "pkcs8", "-topk8", "-nocrypt", "-in", str(tmp / "k.pem")], check=True, capture_output=True).stdout
    assert load_private_key(p8) == load_private_key(pem)


def test_session_gate_then_handshake():
    air = FakeAir()
    key = load_private_key(_test_key_pem())
    with AsiairSession("127.0.0.1", air.port, timeout=0.5) as s:
        assert s.call("test_connection") == "server connected!"
        # the banner and the interleaved PiStatus are events, not replies
        kinds = [e.get("Event") for e in s.events()]
        assert kinds == ["Version", "PiStatus"]
        # unauthenticated: the gate answers with silence -> TimeoutError
        try:
            s.call("get_plan")
            raise AssertionError("gate should have dropped the call")
        except TimeoutError:
            pass
        assert s.verify(key) is True
        assert s.call("get_plan", [7]) == {"echo": [7]}
        try:
            s.call("fails")
            raise AssertionError("device error should raise")
        except AsiairRpcError as e:
            assert e.reply["code"] == 109
    assert [r["method"] for r in air.seen] == [
        "test_connection", "get_plan", "get_verify_str", "verify_client", "pi_is_verified", "get_plan", "fails",
    ]


if __name__ == "__main__":
    assert shutil.which("openssl"), "openssl needed for the signature tests"
    for fn in (test_signature_matches_openssl, test_pkcs8_pem_also_loads, test_session_gate_then_handshake):
        fn()
        print("ok", fn.__name__)
