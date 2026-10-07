"""Talk to an ASIAir's imager port from the shell: read state, capture plan
and sequence JSON, watch events.

    python scripts/asiair_rpc.py 10.0.0.28 get_plan
    python scripts/asiair_rpc.py 10.0.0.28 list_plan get_wheel_setting get_sequence_setting
    python scripts/asiair_rpc.py 10.0.0.28 --listen 60

Read-only by construction: any method that starts with set_/start_/stop_/
import_/clear_/delete_/reset_/scope_/move_/pi_ (other than pi_is_verified)
is refused unless `--allow-write` is given. The handshake runs when a key
file exists (`$ASTROFLOW_ASIAIR_KEY` or the app-support dir); without one
only the ungated methods answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asiair.rpc_session import AsiairRpcError, AsiairSession  # noqa: E402
from asiair.rsa_sign import default_key_path  # noqa: E402

_WRITE = re.compile(r"^(set_|start_|stop_|import_|clear_|delete_|reset_|scope_|move_|open_|close_|pi_(?!is_verified))")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ip")
    ap.add_argument("methods", nargs="*", help="method or method:JSON-params, e.g. get_img_file_info:[0]")
    ap.add_argument("--port", type=int, default=4700)
    ap.add_argument("--key", type=Path, default=None, help=f"PEM key (default {default_key_path()})")
    ap.add_argument("--listen", type=float, default=0, metavar="SECONDS", help="print Events for this long")
    ap.add_argument("--allow-write", action="store_true", help="permit state-changing methods")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--save", type=Path, default=None, metavar="DIR",
                    help="also write each result to DIR/<ip>_<method>_<timestamp>.json (plan backups)")
    a = ap.parse_args()

    refused = [m for m in a.methods if _WRITE.match(m.split(":", 1)[0])]
    if refused and not a.allow_write:
        print(f"refusing state-changing method(s) without --allow-write: {refused}", file=sys.stderr)
        return 2

    key_path = a.key or default_key_path()
    with AsiairSession(a.ip, a.port, timeout=a.timeout) as s:
        print(json.dumps({"test_connection": s.call("test_connection")}))
        if key_path.exists():
            print(json.dumps({"verified": s.verify(key_path)}))
        else:
            print(json.dumps({"verified": False, "note": f"no key at {key_path}; gated methods will time out"}))
        for spec in a.methods:
            method, _, raw = spec.partition(":")
            params = json.loads(raw) if raw else []
            try:
                result = s.call(method, params)
                print(json.dumps({method: result}, indent=2))
                if a.save is not None:
                    a.save.mkdir(parents=True, exist_ok=True)
                    out = a.save / f"{a.ip}_{method}_{time.strftime('%Y%m%d-%H%M%S')}.json"
                    out.write_text(json.dumps(result, indent=2))
                    print(f"saved {out}", file=sys.stderr)
            except (AsiairRpcError, TimeoutError) as e:
                print(json.dumps({method: {"error": str(e)}}))
        if a.listen:
            end = time.time() + a.listen
            while time.time() < end and s.alive:
                for ev in s.events():
                    print(time.strftime("%H:%M:%S"), json.dumps(ev))
                time.sleep(0.2)
        for ev in s.events():
            print("event", json.dumps(ev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
