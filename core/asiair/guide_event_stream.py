"""Persistent listener for an ASIAir's PHD2 guide event stream.

Opens one TCP connection to port 4400 per discovered ASIAir and reads
JSON-RPC notifications. The signal we care about is `GuideStep` — fires
once per guide exposure (~1 Hz) carrying per-axis raw drift in pixels,
SNR, StarMass, and PHD2's own running average distance (`AvgDist`, which
is effectively the recent guide RMS in pixels).

We maintain a rolling buffer (last `_HISTORY_LEN` samples) per rig
(keyed by the device serial) in context so the panel endpoint can render a sparkline. Sky-
quality estimation (computing transparency-loss flags from SNR/StarMass
drops) lives in `asiair.sky_quality` — this module is the raw collector.

Cross-process: stored via `context.*` (actor-backed ContextActor), so
the Flask process reads the same buffer the SchedulerActor wrote.
"""
from __future__ import annotations

import json
import socket
import time
from typing import Any


# Per-host rolling buffer of normalized GuideStep records.
# Key shape: ("asiair_guide_history", rig_key).
# Value shape: list[dict] — list of normalized GuideStep dicts (see
# `_normalize_guide_step`), oldest first.
_HISTORY_LEN = 120  # 2 min at 1 Hz — long enough for RMS baselining.
_HISTORY_KEY_PREFIX = "asiair_guide_history"

# Reconnect/keepalive tuning.
_RECV_TIMEOUT_S = 1.0
_KEEPALIVE_INTERVAL_S = 5.0
_RECONNECT_BACKOFF_S = 2.0


def _history_key(rig_key: str) -> tuple:
    return (_HISTORY_KEY_PREFIX, rig_key)


def _normalize_guide_step(msg: dict, pixel_scale: float | None) -> dict:
    """Project the PHD2 `GuideStep` notification down to the fields the
    UI needs, in canonical units.

    `pixel_scale` (arcsec/px) is read once per ASIAir via
    `get_pixel_scale` and passed in here so we can convert px → arcsec
    server-side; the UI just renders the arcsec series.
    """
    ra_px = msg.get("RADistanceRaw")
    dec_px = msg.get("DECDistanceRaw")
    avg_dist_px = msg.get("AvgDist")

    def _arcsec(v):
        if v is None or pixel_scale is None:
            return None
        return float(v) * pixel_scale

    return {
        "frame": msg.get("Frame"),
        "time": msg.get("Time"),
        "ra_arcsec": _arcsec(ra_px),
        "dec_arcsec": _arcsec(dec_px),
        "rms_arcsec": _arcsec(avg_dist_px),
        "snr": msg.get("SNR"),
        "star_mass": msg.get("StarMass"),
        "ra_pulse_ms": msg.get("RADuration"),
        "dec_pulse_ms": msg.get("DECDuration"),
        "is_dither": bool(msg.get("IsDither")),
        "is_settle": bool(msg.get("IsSettle")),
    }


def _push_to_buffer(rig_key: str, record: dict) -> None:
    """Append `record` to the rolling history for `rig_key`, trim to
    the last `_HISTORY_LEN`. Goes through the actor-backed context so
    Flask workers see the writes."""
    from context import context
    key = _history_key(rig_key)
    prev = context.get(key) or []
    if not isinstance(prev, list):
        prev = []
    prev.append(record)
    if len(prev) > _HISTORY_LEN:
        prev = prev[-_HISTORY_LEN:]
    context[key] = prev


def get_history(rig_key: str) -> list[dict]:
    """Read the rolling buffer for one ASIAir. Empty list when no data
    yet. Called by the panel endpoint."""
    from context import local_context as context
    val = context.get(_history_key(rig_key))
    return val if isinstance(val, list) else []


def _read_pixel_scale(ip: str) -> float | None:
    """One-shot RPC to fetch the guider's pixel_scale (arcsec/px).
    Returns None on any failure — we then store px in the buffer
    without conversion."""
    from asiair.rpc_client import call
    reply = call(ip, 4400, "get_pixel_scale", [], timeout=2.0)
    if reply is None:
        return None
    result = reply.get("result")
    try:
        return float(result)
    except (TypeError, ValueError):
        return None


def stream_one(ip: str, rig_key: str, stop_check=lambda: False) -> None:
    """Blocking event loop for one ASIAir. Opens a persistent TCP
    connection to port 4400, sends periodic keepalive `get_app_state`
    RPCs (without one the imager port idle-closes; the guider seems
    fine without but it's cheap insurance), parses `GuideStep`
    notifications into the rolling buffer.

    `stop_check()` is called periodically; when it returns True the
    loop exits cleanly. Use this from an asyncio task by passing
    `lambda: task.cancelled()` or similar; callers running in plain
    threads can just pass `lambda: False` and use socket close to stop.

    Errors are isolated per-iteration — a malformed line, transient
    disconnect, or recv error logs and reconnects rather than killing
    the loop. The loop is the only thing keeping our buffer fresh; it
    must survive cosmic rays.
    """
    pixel_scale = _read_pixel_scale(ip)
    print(f"### guide_event_stream({rig_key}): pixel_scale={pixel_scale}")

    rpc_id = 1
    sock: socket.socket | None = None
    buf = b""
    next_keepalive = time.time()

    def reconnect() -> bool:
        nonlocal sock, buf, next_keepalive
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        try:
            sock = socket.create_connection((ip, 4400), timeout=2.0)
            sock.settimeout(_RECV_TIMEOUT_S)
            buf = b""
            next_keepalive = time.time()
            return True
        except Exception as e:
            print(f"### guide_event_stream({rig_key}): reconnect failed: {e}")
            return False

    if not reconnect():
        return

    try:
        while not stop_check():
            # Periodic keepalive so the server's idle disconnect doesn't bite.
            if time.time() >= next_keepalive:
                next_keepalive = time.time() + _KEEPALIVE_INTERVAL_S
                try:
                    req = (json.dumps({
                        "id": rpc_id, "method": "get_app_state", "params": []
                    }) + "\r\n").encode()
                    assert sock is not None
                    sock.sendall(req)
                    rpc_id += 1
                except (ConnectionResetError, BrokenPipeError, OSError) as e:
                    print(f"### guide_event_stream({rig_key}): keepalive failed: {e}, reconnecting")
                    if not reconnect():
                        time.sleep(_RECONNECT_BACKOFF_S)
                        continue

            try:
                assert sock is not None
                chunk = sock.recv(8192)
            except socket.timeout:
                continue
            except (ConnectionResetError, OSError) as e:
                print(f"### guide_event_stream({rig_key}): recv error: {e}, reconnecting")
                if not reconnect():
                    time.sleep(_RECONNECT_BACKOFF_S)
                continue

            if not chunk:
                # Peer closed — reconnect.
                print(f"### guide_event_stream({rig_key}): peer closed, reconnecting")
                if not reconnect():
                    time.sleep(_RECONNECT_BACKOFF_S)
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
                event = msg.get("Event")
                if event == "GuideStep":
                    record = _normalize_guide_step(msg, pixel_scale)
                    try:
                        _push_to_buffer(rig_key, record)
                    except Exception as e:
                        print(f"### guide_event_stream({rig_key}): push failed: {e}")
                # StarLost / Settling / Alert events — left as future work
                # (catch them once we see them in the wild and decide
                # exactly how to surface).
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
