"""One-off probe script for surfacing whatever the ASIAir guide loop
exposes on TCP 4400 when guiding is actively running.

Run from `core/` against the live ASIAir while it's guiding:

    python -m asiair.probe_guide 10.0.0.28

Fires a list of candidate JSON-RPC methods sequentially. For each: ✓ if
the device answered with a non-null result, the result is pretty-
printed; ✗ if the call errored or timed out, just the name. Methods
that error are still informative — "method not found" tells us they
don't exist; "params missing" tells us they do but need args we'd then
chase.

Method list is GUESSES — common ASIAir / phd2-ish names plus the few
we already know. The goal is "see what answers"; once we know what
answers, we shape the panel rendering.
"""
from __future__ import annotations

import argparse
import json
import sys

from asiair.rpc_client import call


# PHD2 JSON-RPC method list, lifted from the official PHD2 EventMonitor
# protocol docs. ASIAir's guide engine IS PHD2, so any of these may
# answer.
_PROBE_METHODS: list[tuple[str, list]] = [
    # ---- known hits (sanity-check) ----
    ("get_app_state", []),
    ("get_camera_info", []),
    ("get_setting", []),
    # ---- full PHD2 RPC method list (most important) ----
    ("get_calibrated", []),
    ("get_calibration_data", []),
    ("get_connected", []),
    ("get_cooler_status", []),
    ("get_current_equipment", []),
    ("get_dec_guide_mode", []),
    ("get_exposure", []),
    ("get_exposure_durations", []),
    ("get_lock_position", []),
    ("get_lock_shift_enabled", []),
    ("get_lock_shift_params", []),
    ("get_paused", []),
    ("get_pixel_scale", []),
    ("get_profile", []),
    ("get_profiles", []),
    ("get_search_region", []),
    ("get_sensor_temperature", []),
    ("get_use_subframes", []),
    ("get_algo_param_names", ["RA"]),
    ("get_algo_param_names", ["DEC"]),
    ("get_algo_param", ["RA", "MinMove"]),
    ("get_algo_param", ["DEC", "MinMove"]),
    ("get_variable_delay_settings", []),
    # ---- generic / catch-all ----
    ("get_info", []),
    ("get_status", []),
    ("get_state", []),
    ("get_data", []),
    ("get_all", []),
    ("get_all_state", []),
    ("status", []),
    ("info", []),
    ("query", []),
    ("ping", []),
    # ---- camelCase variants (ZWO SDK convention) ----
    ("getAppState", []),
    ("getGuideState", []),
    ("getGuiderState", []),
    ("getGuideStatus", []),
    ("getGuideData", []),
    ("getGuideRms", []),
    ("getGuideStar", []),
    ("getGuideStep", []),
    ("getGuideHistory", []),
    ("getCalibrationState", []),
    ("getStarPosition", []),
    ("getCameraState", []),
    # ---- prefixed variants ----
    ("guide_state", []),
    ("guide_status", []),
    ("guide_data", []),
    ("guide_info", []),
    ("guide_rms", []),
    ("guide_history", []),
    ("guide_calibration", []),
    ("asiair_state", []),
    ("asiair_status", []),
    ("phd_state", []),
    ("phd_status", []),
    # ---- guider as compound noun ----
    ("guider_info", []),
    ("guider_status", []),
    ("guider_data", []),
    ("guider_state", []),
    # ---- alternate verbs ----
    ("read_app_state", []),
    ("query_app_state", []),
    ("fetch_app_state", []),
    ("read_guide_state", []),
    ("query_guide_state", []),
    ("fetch_guide_state", []),
    ("list_guide_data", []),
    ("show_guide_data", []),
    # ---- pluralized / event-stream style ----
    ("get_guide_events", []),
    ("get_guide_samples", []),
    ("get_guide_points", []),
    ("get_guide_errors", []),
    ("get_guide_deltas", []),
    # ---- subscribe / stream patterns ----
    ("subscribe", []),
    ("subscribe_guide", []),
    ("subscribe_guide_events", []),
    ("start_guide_stream", []),
    # ---- with empty dict params instead of list ----
    # (some JSON-RPC servers require object params)
    ("get_guide_data", {}),
    ("get_guide_state", {}),
    # ---- imager / OTA / mount / focuser surfaces that some libs put
    # adjacent to guide telemetry ----
    ("get_device_state", []),
    ("pi_get_info", []),
    ("get_mount_state", []),
    ("get_mount_info", []),
    ("get_focuser_state", []),
    ("get_focuser_info", []),
    ("get_filter_state", []),
    ("get_filter_info", []),
    # ---- ASIAir-specific names (from asiair-cli / RPC docs) ----
    ("get_app_setting", []),
    ("get_control_status", []),
    ("get_device_info", []),
    ("get_network_state", []),
    ("get_network_info", []),
    ("get_system_info", []),
    ("get_system_state", []),
    ("test_connection", []),
    ("hello", []),
    ("rpcVersion", []),
    ("get_version", []),
    # ---- PHD2 method names (ASIAir's autoguide engine is PHD2-derived) ----
    ("get_app_state2", []),
    ("get_calibrated", []),
    ("get_connected", []),
    ("get_current_equipment", []),
    ("get_dec_guide_mode", []),
    ("get_exposure", []),
    ("get_lock_position", []),
    ("get_lock_shift_enabled", []),
    ("get_lock_shift_params", []),
    ("get_paused", []),
    ("get_pixel_scale", []),
    ("get_profile", []),
    ("get_profiles", []),
    ("get_search_region", []),
    ("get_sensor_temperature", []),
    ("get_star_image", []),
    ("get_use_subframes", []),
    # ---- discovery: just ask what methods exist ----
    ("rpc.discover", []),
    ("system.listMethods", []),
    ("methods", []),
    ("list_methods", []),
    ("help", []),
    ("commands", []),
]


def _pretty(value) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def probe(ip: str, port: int = 4400, timeout: float = 2.0) -> None:
    """Hit every method in `_PROBE_METHODS` once. Print a single
    line per call, plus the pretty-printed result on hits."""
    print(f"### probing {ip}:{port} — {len(_PROBE_METHODS)} methods\n")
    hits: list[tuple[str, dict]] = []
    misses: list[str] = []
    for method, params in _PROBE_METHODS:
        reply = call(ip, port, method, params, timeout=timeout)
        if reply is None:
            misses.append(method)
            print(f"  ✗  {method}")
            continue
        result = reply.get("result")
        hits.append((method, result))
        print(f"  ✓  {method}")
        # Indent the pretty-printed result a touch so hits stand out.
        for line in _pretty(result).splitlines():
            print(f"      {line}")
        print()

    print("\n### summary")
    print(f"  hits ({len(hits)}): {', '.join(m for m, _ in hits) or '(none)'}")
    print(f"  misses ({len(misses)}): {', '.join(misses) or '(none)'}")
    if hits:
        print("\n### next step")
        print("  Use the hit names + result shapes to decide the panel rendering.")
        print("  A time-series field (guide RMS, RA/DEC error) → sparkline path.")
        print("  A single scalar that changes per poll → 1Hz client-side history.")


def listen_for_events(ip: str, port: int = 4400, duration_s: float = 15.0) -> None:
    """Open a persistent connection and read whatever the server pushes
    as async events. PHD2 emits JSON-RPC notifications (no "id" field,
    `Event` string in body) on the same TCP connection between RPC
    responses — most importantly `GuideStep` once per guide exposure,
    carrying per-axis raw drift in pixels.

    Just opens the socket, reads lines for `duration_s` seconds, prints
    each as it arrives. Categorises by `Event` name at the end.
    """
    import socket
    import collections

    print(f"\n### listening for events on {ip}:{port} for {duration_s}s\n")
    seen: dict[str, int] = collections.Counter()
    samples: dict[str, dict] = {}

    sock = socket.create_connection((ip, port), timeout=2.0)
    sock.settimeout(2.0)
    buf = b""
    deadline = duration_s
    import time
    t0 = time.time()
    try:
        while time.time() - t0 < deadline:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                print("  (peer closed connection)")
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    print(f"  ?  unparseable: {line[:80]!r}")
                    continue
                if not isinstance(msg, dict):
                    continue
                event = msg.get("Event") or msg.get("event")
                if event:
                    seen[event] += 1
                    samples.setdefault(event, msg)
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:5.1f}s]  {event}")
    finally:
        sock.close()

    print("\n### event summary")
    if not seen:
        print("  (no events received — guide loop may be paused, or events live on a different port)")
        return
    for ev, count in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d} × {ev}")
    print("\n### sample of each event type")
    for ev, sample in sorted(samples.items()):
        print(f"\n--- {ev} ---")
        print(_pretty(sample))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ip", help="ASIAir device IP (e.g., 10.0.0.28)")
    p.add_argument(
        "--port", type=int, default=4400,
        help="TCP port (default 4400 = guider port)",
    )
    p.add_argument(
        "--timeout", type=float, default=2.0,
        help="Per-call timeout seconds (default 2.0)",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    probe(args.ip, port=args.port, timeout=args.timeout)
    listen_for_events(args.ip, port=args.port, duration_s=15.0)
