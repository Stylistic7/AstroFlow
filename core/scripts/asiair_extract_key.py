"""Pull the ASIAir handshake key out of ZWO's Android app package.

    python scripts/asiair_extract_key.py ~/Downloads/ASIAIR_3.0.apk

ZWO hosts the APK itself (zwoastro.com/software/asiair, "Android" link).
The key is a plain PEM block inside the package's `libopenssllib.so`; this
walks every nested zip (APK/XAPK/split APKs), finds the first PRIVATE KEY
block, checks it parses, and writes it with mode 600 to the default key
path (`$ASTROFLOW_ASIAIR_KEY` or the app-support dir). The package itself
is read and never kept.
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asiair.rsa_sign import default_key_path, load_private_key  # noqa: E402

_PEM = re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.S)


def _walk(data: bytes, name: str):
    """Yield (name, bytes) for every leaf file inside nested zips."""
    if data[:2] != b"PK":
        yield name, data
        return
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        yield name, data
        return
    for info in zf.infolist():
        if info.is_dir():
            continue
        yield from _walk(zf.read(info), f"{name}/{info.filename}")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    package = Path(sys.argv[1]).expanduser()
    rejected = []
    for name, data in _walk(package.read_bytes(), package.name):
        for m in _PEM.finditer(data):
            pem = m.group(0) + b"\n"
            try:
                key = load_private_key(pem)
            except Exception as e:  # a template string, a non-RSA key, a truncated block
                rejected.append((name, len(pem), f"{type(e).__name__}: {e}"))
                continue
            out = default_key_path()
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(pem)
            out.chmod(0o600)
            print(f"found a {key.n.bit_length()}-bit RSA key in {name}\nwritten to {out} (mode 600)")
            return 0
    print("no usable PRIVATE KEY block found in the package", file=sys.stderr)
    for name, size, why in rejected:
        print(f"  rejected block in {name} ({size} bytes): {why}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
