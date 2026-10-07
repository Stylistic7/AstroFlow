"""RSA PKCS#1 v1.5 / SHA-1 signing in pure Python — the ASIAir's 4700 handshake.

Firmware 7.18+ gates nearly every imager-port method behind a challenge:
`get_verify_str` returns a random string, the client signs it with the RSA
private key embedded in ZWO's own app, and `verify_client([signature,
challenge])` unlocks the connection. The signature is textbook PKCS#1 v1.5
over SHA-1, base64-encoded.

Pure Python on purpose: the shared venv must not grow a `cryptography`
dependency for a few dozen lines of arithmetic, and `pow(m, d, n)` on a
2048-bit key takes ~10 ms. Only what the handshake needs is implemented —
PKCS#1 and PKCS#8 PEM parsing, no encryption, no key generation.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from base_path import app_support_dir

# DER prefix of DigestInfo for SHA-1 (RFC 8017 §9.2, note 1).
_SHA1_DIGEST_INFO = bytes.fromhex("3021300906052b0e03021a05000414")
# PKCS#8 wraps the PKCS#1 key as an OCTET STRING after an rsaEncryption AlgorithmIdentifier.
_RSA_OID = bytes.fromhex("06092a864886f70d010101")
_PEM_BLOCK = re.compile(
    rb"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY)-----(?P<body>.*?)-----END (?P=label)-----",
    re.S,
)

KEY_ENV = "ASTROFLOW_ASIAIR_KEY"


@dataclass(frozen=True)
class RsaPrivateKey:
    n: int
    d: int

    @property
    def size_bytes(self) -> int:
        return (self.n.bit_length() + 7) // 8


def default_key_path() -> Path:
    """Where the extracted key lives: `$ASTROFLOW_ASIAIR_KEY` or the app
    support dir. Never inside the repo."""
    env = os.environ.get(KEY_ENV)
    return Path(env).expanduser() if env else app_support_dir() / "asiair_key.pem"


def _der_read(buf: bytes, pos: int) -> tuple[int, bytes, int]:
    """One TLV at `pos` -> (tag, value, next_pos). Definite lengths only."""
    tag = buf[pos]
    length = buf[pos + 1]
    pos += 2
    if length & 0x80:
        n = length & 0x7F
        length = int.from_bytes(buf[pos:pos + n], "big")
        pos += n
    return tag, buf[pos:pos + length], pos + length


def _der_sequence(buf: bytes) -> list[tuple[int, bytes]]:
    tag, body, _ = _der_read(buf, 0)
    if tag != 0x30:
        raise ValueError(f"expected SEQUENCE, got tag 0x{tag:02x}")
    items, pos = [], 0
    while pos < len(body):
        tag, value, pos = _der_read(body, pos)
        items.append((tag, value))
    return items


def _pkcs1_key(der: bytes) -> RsaPrivateKey:
    # RSAPrivateKey ::= SEQUENCE { version, n, e, d, p, q, dP, dQ, qInv }
    items = _der_sequence(der)
    if len(items) < 4 or any(tag != 0x02 for tag, _ in items[:4]):
        raise ValueError("not a PKCS#1 RSAPrivateKey")
    n = int.from_bytes(items[1][1], "big")
    d = int.from_bytes(items[3][1], "big")
    return RsaPrivateKey(n=n, d=d)


def load_private_key(pem: bytes | str) -> RsaPrivateKey:
    """Parse the first private-key PEM block in `pem` (PKCS#1 "RSA PRIVATE
    KEY" or PKCS#8 "PRIVATE KEY" wrapping an RSA key)."""
    if isinstance(pem, str):
        pem = pem.encode()
    m = _PEM_BLOCK.search(pem)
    if m is None:
        raise ValueError("no PRIVATE KEY PEM block found")
    der = base64.b64decode(b"".join(m.group("body").split()))
    if m.group("label") == b"RSA PRIVATE KEY":
        return _pkcs1_key(der)
    # PrivateKeyInfo ::= SEQUENCE { version, AlgorithmIdentifier, OCTET STRING privateKey }
    items = _der_sequence(der)
    if len(items) < 3 or _RSA_OID not in items[1][1] or items[2][0] != 0x04:
        raise ValueError("PKCS#8 block is not an RSA key")
    return _pkcs1_key(items[2][1])


def load_key_file(path: Path | None = None) -> RsaPrivateKey:
    return load_private_key(Path(path or default_key_path()).read_bytes())


def sign_sha1_pkcs1v15(key: RsaPrivateKey, message: bytes) -> bytes:
    """RFC 8017 RSASSA-PKCS1-v1_5 with SHA-1: EM = 00 01 FF..FF 00 DigestInfo."""
    digest_info = _SHA1_DIGEST_INFO + hashlib.sha1(message).digest()
    k = key.size_bytes
    pad_len = k - len(digest_info) - 3
    if pad_len < 8:
        raise ValueError("key too short for PKCS#1 v1.5 padding")
    em = b"\x00\x01" + b"\xff" * pad_len + b"\x00" + digest_info
    s = pow(int.from_bytes(em, "big"), key.d, key.n)
    return s.to_bytes(k, "big")


def sign_challenge(key: RsaPrivateKey, challenge: str) -> str:
    """The exact string `verify_client` expects: base64 of the signature
    over the challenge's UTF-8 bytes."""
    return base64.b64encode(sign_sha1_pkcs1v15(key, challenge.encode())).decode()
