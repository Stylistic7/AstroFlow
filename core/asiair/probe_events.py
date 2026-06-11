"""Deep event-stream capture from both ASIAir ports simultaneously.

Opens persistent TCP connections to the guider (4400) and imager (4700)
ports and reads JSON-RPC notifications for `duration_s` seconds.
Categorises every distinct `Event` name, counts occurrences, and prints
one representative sample of each.

Usage:
    python -m asiair.probe_events 10.0.0.28 --duration 60
"""
from __future__ import annotations

import argparse
import collections
import json
import socket
import sys
import time
from threading import Thread


def listen(ip: str, port: int, duration_s: float, label: str,
           events_by_port: dict, samples_by_port: dict,
           keepalive_method: str | None = "get_app_state",
           keepalive_interval_s: float = 5.0) -> None:
    """Connect, read JSON-line events for duration_s, record into shared dicts.

    Sends a periodic `keepalive_method` RPC so the server doesn't idle-
    disconnect us (the imager port closes after ~15s of silence). Tolerates
    transient disconnects with auto-reconnect."""
    seen: collections.Counter = collections.Counter()
    samples: dict = {}
    t0 = time.time()
    next_keepalive = t0
    rpc_id = 1
    sock: socket.socket | None = None
    buf = b""

    def reconnect():
        nonlocal sock, buf
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        try:
            sock = socket.create_connection((ip, port), timeout=2.0)
            sock.settimeout(1.0)
            buf = b""
            return True
        except Exception as e:
            print(f"[{label}:{port}] reconnect failed: {e}")
            return False

    if not reconnect():
        events_by_port[port] = seen
        samples_by_port[port] = samples
        return

    try:
        while time.time() - t0 < duration_s:
            # Keepalive
            if keepalive_method and time.time() >= next_keepalive:
                next_keepalive = time.time() + keepalive_interval_s
                try:
                    req = (json.dumps({
                        "id": rpc_id, "method": keepalive_method, "params": []
                    }) + "\r\n").encode()
                    sock.sendall(req)
                    rpc_id += 1
                except (ConnectionResetError, BrokenPipeError, OSError):
                    if not reconnect():
                        time.sleep(1.0)
                        continue
            try:
                chunk = sock.recv(8192)
            except socket.timeout:
                continue
            except (ConnectionResetError, OSError) as e:
                print(f"[{label}:{port}] recv error at t={time.time()-t0:.1f}s: {e}")
                if not reconnect():
                    time.sleep(1.0)
                continue
            if not chunk:
                print(f"[{label}:{port}] peer closed at t={time.time()-t0:.1f}s, reconnecting")
                if not reconnect():
                    time.sleep(1.0)
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                event = msg.get("Event") or msg.get("event") or msg.get("type")
                if event:
                    seen[event] += 1
                    samples[event] = msg
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
    events_by_port[port] = seen
    samples_by_port[port] = samples
    print(f"[{label}:{port}] done — {sum(seen.values())} events, "
          f"{len(seen)} distinct types: {', '.join(sorted(seen))}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ip")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Seconds to listen (default 60)")
    p.add_argument("--ports", nargs="+", type=int, default=[4400, 4700],
                   help="Ports to listen on (default 4400 4700)")
    args = p.parse_args()

    print(f"### listening on {args.ip} ports {args.ports} for {args.duration}s\n")

    events_by_port: dict = {}
    samples_by_port: dict = {}
    threads: list[Thread] = []
    for port in args.ports:
        label = {4400: "guider", 4700: "imager"}.get(port, str(port))
        t = Thread(
            target=listen,
            args=(args.ip, port, args.duration, label,
                  events_by_port, samples_by_port),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("\n\n### full event sample dump")
    for port in sorted(events_by_port.keys()):
        label = {4400: "guider", 4700: "imager"}.get(port, str(port))
        print(f"\n========== {label} port {port} ==========")
        events = events_by_port[port]
        samples = samples_by_port[port]
        if not events:
            print("  (no events received)")
            continue
        for ev, count in sorted(events.items(), key=lambda kv: -kv[1]):
            print(f"\n--- [{count}×] {ev} ---")
            print(json.dumps(samples[ev], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
