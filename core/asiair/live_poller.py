"""Finds the ASIAirs on the LAN and polls each for live state.

Discovery is a port-4700 scan of the directly attached networks, not
mDNS: firmware 14.39 announces every unit as `asiair.local`, so Bonjour
listed ONE rig however many were powered on (2026-09-05, 2026-10-07).
Identity is the device serial — `get_device_state.device.sn`, which the
imager port tells only a key-verified session — so a rig keeps its
record through failed polls and DHCP moves, and an address that answers
with a known serial updates that rig instead of adding one.

`context.asiair_devices` is the rig list, written here and read by the
panel, the guide-event streams and the downloader:

    {"key": sn (the IP when there is no key file),
     "ip", "sn", "name": ap_ssid, "focal_len_mm",
     "live": {queried_at, reachable, sn, ap_ssid, firmware, model, pi_temp_c,
              focal_len_mm, location, focuser_state, camera_state,
              camera_name, last_goto_target, capture_state, stack_total,
              stack_done, guide_state, guide_app_state_raw}}

A field a call did not answer stays missing; a cycle in which nothing
answered sets `reachable=False` and the row goes gray.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import structlog

from asiair.network_hosts import find_hosts_with_open_port
from asiair.rpc_session import GUIDER_PORT, IMAGER_PORT, AsiairRpcError, AsiairSession
from asiair.rsa_sign import RsaPrivateKey, default_key_path, load_key_file

log = structlog.get_logger()

_IMAGER_METHODS = ("get_device_state", "get_app_state", "get_app_setting", "pi_get_info",
                   "get_camera_state", "get_focuser_state")
_SCAN_INTERVAL_S = 120.0
_next_scan = 0.0
_warned_no_key = False


def _key() -> RsaPrivateKey | None:
    """The signing key, or None (warned once) when the file is absent: the
    imager port then answers nothing and rigs are keyed by IP."""
    global _warned_no_key
    path = default_key_path()
    if path.exists():
        return load_key_file(path)
    if not _warned_no_key:
        log.warning("asiair: no key file, rigs are identified by IP only", path=str(path))
        _warned_no_key = True
    return None


def _normalize_guide_state(raw: Any) -> str | None:
    """The guider port's `get_app_state`: the bare string "Stopped" when
    idle; a dict with a state field once a loop is running."""
    if isinstance(raw, str):
        return raw.strip().lower() or None
    if isinstance(raw, dict):
        for key in ("state", "guide_state", "status", "mode"):
            v = raw.get(key)
            if isinstance(v, str) and v:
                return v.strip().lower()
    return None


def _build_live_snapshot(results: dict[str, Any]) -> dict[str, Any]:
    """Flatten the per-method results into the snapshot. `results` holds
    only the calls that answered, so an empty dict is "unreachable"."""
    out: dict[str, Any] = {"queried_at": time.time(), "reachable": bool(results)}
    device_state = results.get("get_device_state") or {}
    device = device_state.get("device") or {}
    pi_info = results.get("pi_get_info") or {}
    app_setting = results.get("get_app_setting") or {}
    app_state = results.get("get_app_state") or {}
    fields = {
        "sn": device.get("sn") or pi_info.get("guid"),
        "ap_ssid": (device_state.get("ap") or {}).get("ssid"),
        "firmware": device.get("firmware_ver_string"),
        "focal_len_mm": device.get("focal_len"),
        "location": device_state.get("location_lon_lat"),
        "focuser_state": (device_state.get("focuser") or {}).get("state")
        or (results.get("get_focuser_state") or {}).get("state"),
        "model": pi_info.get("model"),
        "pi_temp_c": pi_info.get("temp"),
        "last_goto_target": app_setting.get("goto_target_name"),
        "camera_name": app_setting.get("main_camera_name"),
        "camera_state": (results.get("get_camera_state") or {}).get("state"),
        "capture_state": (app_state.get("capture") or {}).get("state"),
        "stack_total": (app_state.get("stack") or {}).get("total_frame"),
        "stack_done": (app_state.get("stack") or {}).get("stacked_frame"),
    }
    out.update({k: v for k, v in fields.items() if v not in (None, "")})
    if "guide_app_state" in results:
        out["guide_app_state_raw"] = results["guide_app_state"]
        if state := _normalize_guide_state(results["guide_app_state"]):
            out["guide_state"] = state
    return out


def poll_ip(ip: str, key: RsaPrivateKey | None, timeout: float = 2.0) -> dict[str, Any]:
    """One poll of one address: the verified imager session for identity
    and capture state, the guider session for the loop state. A call that
    fails leaves its slice missing; a port that fails leaves all of its."""
    results: dict[str, Any] = {}
    if key is not None:
        try:
            with AsiairSession(ip, IMAGER_PORT, timeout=timeout) as s:
                if not s.verify(key):
                    log.warning("asiair poll: key rejected", ip=ip)
                for method in _IMAGER_METHODS if s.verified else ():
                    try:
                        results[method] = s.call(method)
                    except AsiairRpcError as e:
                        log.debug("asiair poll", ip=ip, method=method, error=str(e))
        except (OSError, AsiairRpcError) as e:
            log.debug("asiair poll: imager port", ip=ip, error=str(e))
    try:
        with AsiairSession(ip, GUIDER_PORT, timeout=timeout) as s:
            results["guide_app_state"] = s.call("get_app_state")
    except (OSError, AsiairRpcError) as e:
        log.debug("asiair poll: guider port", ip=ip, error=str(e))
    return _build_live_snapshot(results)


def _merge(rigs: list[dict], polled: list[tuple[str, dict]]) -> list[dict]:
    """Fold one cycle's snapshots into the rig list. A snapshot with a
    serial finds (or re-keys) the rig of that serial; without one it
    updates the rig last seen at that address, or — when something did
    answer — starts an IP-keyed rig. Silence from an unknown address is
    nothing to record."""
    by_key = {r["key"]: dict(r) for r in rigs}
    for ip, live in polled:
        prev = next((r for r in by_key.values() if r["ip"] == ip), None)
        if prev is None and "sn" not in live and not live.get("reachable"):
            continue
        key = live.get("sn") or (prev["key"] if prev else ip)
        if prev is not None and prev["key"] != key:
            del by_key[prev["key"]]
        rec = by_key.get(key) or prev or {}
        rec.update(key=key, ip=ip, live=live)
        for field, src in (("sn", "sn"), ("name", "ap_ssid"), ("focal_len_mm", "focal_len_mm")):
            if src in live:
                rec[field] = live[src]
        by_key[key] = rec
    return sorted(by_key.values(), key=lambda r: r["key"])


def tick(*, scan: Callable[[int], list[str]] = find_hosts_with_open_port,
         poll: Callable[[str, RsaPrivateKey | None], dict] = poll_ip,
         now: Callable[[], float] = time.time) -> None:
    """One cycle: poll every known address and, every `_SCAN_INTERVAL_S`,
    every host with the imager port open; write the rig list back and
    tell the web clients. The polls run in parallel so a powered-off rig's
    connect timeout never delays a live one."""
    from context import context  # local import: avoids a cycle at module load
    global _next_scan
    rigs = list(context.asiair_devices or [])
    ips = {r["ip"] for r in rigs}
    if now() >= _next_scan:
        ips |= set(scan(IMAGER_PORT))
        _next_scan = now() + _SCAN_INTERVAL_S
    if not ips:
        return
    key = _key()
    with ThreadPoolExecutor(max_workers=len(ips)) as ex:
        polled = list(zip(sorted(ips), ex.map(lambda ip: poll(ip, key), sorted(ips))))
    context.asiair_devices = _merge(rigs, polled)
    from web_server.backend.announcer import announce
    announce("asiair_devices")
