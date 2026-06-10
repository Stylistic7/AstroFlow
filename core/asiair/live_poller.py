"""Periodic poller that pulls live state from every discovered ASIAir
and merges the result into `context.asiair_devices[i]["live"]`.

Runs alongside the existing SMB downloader in the SchedulerActor's
async loop. One pass per tick; per-device parallelism kept implicit
(the cost is dominated by 6 short LAN RPC calls, ~30-60ms total per
device — fine to serialize).

State written to `context.asiair_devices[i]["live"]`:

```
"live": {
    "queried_at": float,      # epoch seconds of the last cycle
    "ap_ssid": str,           # user-set device name ("ASIAIR_RC71")
    "firmware": str,          # "13.41"
    "model": str,             # "ZWO AirPlus-RK3568 (Linux)"
    "pi_temp_c": float,       # Pi CPU temp (NOT cooler)
    "focal_len_mm": int,
    "location": [lon, lat] | None,
    "focuser_state": str,     # "open" | "close"
    "camera_state": str,      # "open" | "close"
    "camera_name": str | None,
    "last_goto_target": str | None,
    "capture_state": str,     # from get_app_state.capture.state
    "stack_total": int,       # frames in current stack request
    "stack_done": int,        # frames stacked so far
    "reachable": bool,        # any of the above succeeded
}
```

Any individual call that fails leaves its slice missing — the panel
checks per-field and renders only what's there. A FULLY failed cycle
sets `reachable=False` and dims the panel row to gray.
"""
from __future__ import annotations

import time
import structlog
from typing import Any

from asiair.rpc_client import call_many


log = structlog.get_logger()


# Method list for one poll cycle. All on the imager port (4700) —
# the guider port (4400) didn't expose anything we surface today.
_POLL_METHODS = [
    ("get_device_state", []),
    ("get_app_state", []),
    ("get_app_setting", []),
    ("pi_get_info", []),
    ("get_camera_state", []),
    ("get_focuser_state", []),
]


def _result(reply: dict | None) -> dict | None:
    """Pluck the `.result` payload from a successful JSON-RPC reply,
    or None if the call failed. The reply shape is the post-error-
    filter contract from `rpc_client.call`: non-None means code==0
    and `result` is present."""
    if reply is None:
        return None
    return reply.get("result")


def _build_live_snapshot(replies: dict[str, dict | None]) -> dict[str, Any]:
    """Flatten the per-method results into the panel-friendly snapshot
    shape. Missing fields stay missing — never substitute defaults
    that would lie about state."""
    out: dict[str, Any] = {"queried_at": time.time(), "reachable": False}

    device_state = _result(replies.get("get_device_state"))
    if device_state:
        out["reachable"] = True
        ap = device_state.get("ap") or {}
        if ap.get("ssid"):
            out["ap_ssid"] = ap["ssid"]
        device = device_state.get("device") or {}
        if device.get("firmware_ver_string"):
            out["firmware"] = device["firmware_ver_string"]
        if device.get("focal_len"):
            out["focal_len_mm"] = device["focal_len"]
        loc = device_state.get("location_lon_lat")
        if isinstance(loc, list) and len(loc) == 2:
            out["location"] = list(loc)
        focuser = device_state.get("focuser") or {}
        if focuser.get("state"):
            out["focuser_state"] = focuser["state"]

    pi_info = _result(replies.get("pi_get_info"))
    if pi_info:
        out["reachable"] = True
        if "temp" in pi_info:
            out["pi_temp_c"] = pi_info["temp"]
        if pi_info.get("model"):
            out["model"] = pi_info["model"]

    app_setting = _result(replies.get("get_app_setting"))
    if app_setting:
        out["reachable"] = True
        if app_setting.get("goto_target_name"):
            out["last_goto_target"] = app_setting["goto_target_name"]
        if app_setting.get("main_camera_name"):
            out["camera_name"] = app_setting["main_camera_name"]

    app_state = _result(replies.get("get_app_state"))
    if app_state:
        out["reachable"] = True
        capture = app_state.get("capture") or {}
        if capture.get("state"):
            out["capture_state"] = capture["state"]
        stack = app_state.get("stack") or {}
        if "total_frame" in stack:
            out["stack_total"] = stack["total_frame"]
        if "stacked_frame" in stack:
            out["stack_done"] = stack["stacked_frame"]

    camera_state = _result(replies.get("get_camera_state"))
    if camera_state and camera_state.get("state"):
        out["reachable"] = True
        out["camera_state"] = camera_state["state"]

    focuser_state = _result(replies.get("get_focuser_state"))
    if focuser_state and focuser_state.get("state"):
        out["reachable"] = True
        # `get_device_state.focuser.state` covers this when the
        # device-state call succeeds; this is a fallback for when
        # only the focuser-specific call gets through.
        out.setdefault("focuser_state", focuser_state["state"])

    return out


def poll_device(device: dict[str, Any], port: int = 4700,
                timeout: float = 2.0) -> dict[str, Any]:
    """Run one poll cycle against a single discovered device and
    return the snapshot dict. The caller merges this into
    `context.asiair_devices[i]["live"]`."""
    ip = device.get("ip")
    if not ip:
        return {"queried_at": time.time(), "reachable": False}
    replies = call_many(ip, port, _POLL_METHODS, timeout=timeout)
    return _build_live_snapshot(replies)


def tick() -> None:
    """One poll cycle across all discovered devices. Writes back the
    updated `context.asiair_devices` list and announces an
    `asiair_devices` SSE event so connected clients refetch.

    Best-effort: a probe failure logs at debug and dims that one row;
    a fatal error in this function is caught at the SchedulerActor
    level so the rest of the poll loop keeps going.
    """
    from context import context  # local import: avoids cycle at module load
    devices = list(context.asiair_devices or [])
    if not devices:
        return

    changed = False
    for d in devices:
        try:
            d["live"] = poll_device(d)
            changed = True
        except Exception as e:
            log.warning("asiair live-poller: device crash", ip=d.get("ip"), error=str(e))
    if not changed:
        return
    context.asiair_devices = devices

    try:
        from web_server.backend.announcer import announce
        announce("asiair_devices")
    except Exception as e:
        log.debug("asiair live-poller: announce failed", error=str(e))
