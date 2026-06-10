"""Aggregator for the ASIAir devices dropdown panel.

ONE ROW PER DISCOVERED ASIAir DEVICE. The panel's job is to give the
user comfort that their rig is up and working — capturing the planned
target, guiding, making frame-count progress — without having to also
open the ZWO mobile app. The cross-reference back to AstroFlow's
camera-fingerprint identity (used for calibration-frame association)
is incidental and only surfaces when it's unambiguous (single
discovered ASIAir + single anchored camera in the stacks dict);
deriving per-ASIAir camera attribution from downloaded files would
require download-path changes we don't want here.

Live state comes from `core/asiair/live_poller.py` (writes the live
poll snapshot into `device["live"]`). Files-on-disk state (camera
hash, last-image mtime) comes from `context.stacks` when the
attribution is safe.
"""
from __future__ import annotations

from typing import Any

from context import local_context as context


_ACTIVE_GREEN_SECONDS = 30 * 60        # < 30 min
_ACTIVE_YELLOW_SECONDS = 24 * 60 * 60  # < 24 h


# Guider states the panel renders as "actively guiding (good)".
# Anything else AND capture-state-active is the warning condition
# (the cloudy / star-lost / equipment-disconnected scenario).
_GUIDE_HEALTHY_STATES = {"guiding", "looping", "calibrating", "settling"}


def _derive_guide_warning(live: dict | None) -> bool:
    """The user-facing warning we surface: capture is active on the
    imager port but the guide loop is NOT in a healthy state. That's
    the situation worth surfacing with a ⚠️ icon — clouds, star lost,
    guide cam disconnect mid-session.

    When the imager is idle, a stopped guider is the EXPECTED state
    and we say nothing. When the guider state is unknown (`None`),
    we also stay silent — better than firing spurious alarms while
    we're still learning the firmware's state vocabulary.
    """
    if not live:
        return False
    capture_state = live.get("capture_state")
    guide_state = live.get("guide_state")
    # `capture_state` is from the imager's `get_app_state.capture.state`.
    # "idle" is the not-currently-imaging value; anything else means
    # the user told the device to capture.
    if not capture_state or capture_state == "idle":
        return False
    if guide_state is None:
        return False
    return guide_state not in _GUIDE_HEALTHY_STATES


def _band(epoch: float, now_epoch: float) -> str:
    """Classify a file's recency into the three-state status dot:
    green if newer than 30 min, yellow if newer than 24 h, gray else.
    `epoch=0` is the sentinel for "no files yet" — always gray.
    """
    if epoch <= 0:
        return "gray"
    delta = now_epoch - epoch
    if delta < _ACTIVE_GREEN_SECONDS:
        return "green"
    if delta < _ACTIVE_YELLOW_SECONDS:
        return "yellow"
    return "gray"


def _resolve_stack(stack_or_ref) -> Any:
    """`context.stacks` is a `dict[stable_hash, ContextRef[Stack]]` in
    production, but tests sometimes pass plain Stack objects. Accept
    either."""
    get = getattr(stack_or_ref, "get", None)
    if callable(get):
        return get()
    return stack_or_ref


def _dedup_asiairs_by_hostname(asiairs: list[dict]) -> dict[str, dict]:
    """Discovery often returns the SAME physical ASIAir under multiple
    records — one per Bonjour service type (_smb._tcp + _device-info._tcp),
    or one per network interface (a routable LAN IP plus a link-local
    169.254/16 self-assigned address). Group by mDNS hostname; within
    each hostname, prefer the entry whose live poll succeeded so the
    panel renders state from the IP that actually answered.
    """
    by_host: dict[str, dict] = {}
    for d in asiairs:
        host = d.get("hostname") or d.get("ip") or ""
        if not host:
            continue
        prev = by_host.get(host)
        if prev is None:
            by_host[host] = d
            continue
        prev_reachable = bool((prev.get("live") or {}).get("reachable"))
        this_reachable = bool((d.get("live") or {}).get("reachable"))
        if this_reachable and not prev_reachable:
            by_host[host] = d
    return by_host


def _unambiguous_camera_hash(stacks_dict: dict, n_asiairs: int) -> str | None:
    """If we have exactly one discovered ASIAir AND exactly one
    anchored camera (one unique camera_id_hash across all stacks),
    show it. Anything more complex would need per-ASIAir attribution
    in the download path, which is explicit out — the camera hash is
    low-priority and the wiring isn't worth it."""
    if n_asiairs != 1:
        return None
    hashes: set[str] = set()
    for s_or_ref in stacks_dict.values():
        stack = _resolve_stack(s_or_ref)
        h = getattr(stack, "camera_id_hash", None)
        if h:
            hashes.add(h)
    if len(hashes) == 1:
        return next(iter(hashes))
    return None


def _row_active_state(
    reachable: bool, capture_state: str | None, guide_warning: bool,
) -> str:
    """Four-state status dot for the row.
      - green:   reachable, capturing, guider healthy. The user's
                 "everything's working" state.
      - warning: reachable, capturing, but `guide_warning` fires —
                 the guider stopped while the imager kept going.
                 Equivalent to the chip-level amber. Capturing IS
                 still happening; the frames may just be worse.
      - yellow:  reachable but idle (powered on, no session active).
      - gray:    didn't answer the last poll cycle. We tried; nothing.

    A `red` band ("capture stopped unexpectedly") needs a history of
    prior states to differentiate a clean session end from a mid-
    session crash, and we don't track that yet — see the open issue
    in the conversation log.
    """
    if not reachable:
        return "gray"
    if capture_state and capture_state != "idle":
        return "warning" if guide_warning else "green"
    return "yellow"


def compute_panel_state(now_epoch: float | None = None) -> dict[str, Any]:
    """Build the panel payload from current context. Pure function over
    context state — safe to call from a Flask handler on every GET.

    `now_epoch` is injectable for deterministic tests; defaults to
    `time.time()` at call.
    """
    import time
    if now_epoch is None:
        now_epoch = time.time()

    asiairs = list(context.asiair_devices or [])
    stacks_dict = context.stacks or {}

    by_host = _dedup_asiairs_by_hostname(asiairs)
    cam_hash = _unambiguous_camera_hash(stacks_dict, len(by_host))

    rigs: list[dict[str, Any]] = []
    for host, dev in by_host.items():
        live = dev.get("live") or {}
        reachable = bool(live.get("reachable"))
        capture_state = live.get("capture_state")
        guide_warning = _derive_guide_warning(live or None)
        label = (
            live.get("ap_ssid")
            or host.rstrip(".").removesuffix(".local")
            or dev.get("ip", "")
        )
        rigs.append({
            "key": host,
            "label": label,
            "hostname": host,
            "ip": dev.get("ip", ""),
            "reachable": reachable,
            "active_state": _row_active_state(reachable, capture_state, guide_warning),
            "capture_state": capture_state,
            "stack_done": live.get("stack_done"),
            "stack_total": live.get("stack_total"),
            "last_goto_target": live.get("last_goto_target"),
            "camera_name": live.get("camera_name"),
            "camera_state": live.get("camera_state"),
            "guide_state": live.get("guide_state"),
            "guide_warning": guide_warning,
            # Unambiguous camera-hash carry — see `_unambiguous_camera_hash`
            # for the rules. Empty string when ambiguous so the UI can
            # treat "no hash" identically across the missing / ambiguous
            # cases.
            "camera_hash_short": cam_hash[:8] if cam_hash else "",
            "live": live or None,
        })

    active = sum(1 for r in rigs if r["active_state"] == "green")
    warnings = sum(1 for r in rigs if r.get("guide_warning"))
    return {
        "rigs": rigs,
        "summary": {
            "total": len(rigs),
            "active": active,
            "warnings": warnings,
        },
    }
