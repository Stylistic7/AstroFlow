"""The ASIAir devices dropdown: one row per rig in `context.asiair_devices`
(written by `asiair.live_poller`), the live snapshot reduced to the states
the panel colours, plus the guide-event history and the sky-quality
estimate derived from it."""
from __future__ import annotations

from typing import Any

from context import local_context as context


# Guider states the panel renders as "actively guiding (good)".
_GUIDE_HEALTHY_STATES = {"guiding", "looping", "calibrating", "settling"}


def _derive_guide_warning(live: dict) -> bool:
    """Capture is active on the imager but the guide loop is NOT healthy —
    clouds, star lost, guide-camera disconnect mid-session. An idle imager
    with a stopped guider is the expected daytime state; an unknown guide
    state stays silent rather than alarm on vocabulary we have not seen."""
    capture_state = live.get("capture_state")
    guide_state = live.get("guide_state")
    if not capture_state or capture_state == "idle" or guide_state is None:
        return False
    return guide_state not in _GUIDE_HEALTHY_STATES


def _row_active_state(reachable: bool, capture_state: str | None, guide_warning: bool) -> str:
    """green: capturing, guider healthy. warning: capturing, guider not.
    yellow: reachable but idle. gray: did not answer the last poll."""
    if not reachable:
        return "gray"
    if capture_state and capture_state != "idle":
        return "warning" if guide_warning else "green"
    return "yellow"


def label(rig: dict) -> str:
    """What the user sees: the name set in the ZWO app and the focal length
    of the train it drives — two rigs on the same camera model are told
    apart by their optics."""
    name = rig.get("name") or rig.get("ip") or rig["key"]
    focal = rig.get("focal_len_mm")
    return f"{name} · {focal:g} mm" if focal else name


def compute_panel_state() -> dict[str, Any]:
    """The panel payload: pure over context, called on every GET."""
    from asiair.guide_event_stream import get_history
    from asiair.sky_quality import estimate as estimate_sky

    cooling_requests = context.get("asiair_cooling", None) or {}
    rigs: list[dict[str, Any]] = []
    for rig in sorted(context.asiair_devices or [], key=label):
        live = rig.get("live") or {}
        reachable = bool(live.get("reachable"))
        capture_state = live.get("capture_state")
        guide_warning = _derive_guide_warning(live)
        guide_history = get_history(rig["key"])
        rigs.append({
            "key": rig["key"],
            "label": label(rig),
            "ip": rig.get("ip", ""),
            "sn": rig.get("sn"),
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
            "live": live or None,
            # The slow-ramp request (switch + target) and its live progress.
            "cooling": {"enabled": False, "target_c": None,
                        **(cooling_requests.get(rig["key"]) or {}), **(rig.get("cooling") or {})},
            "guide_history": guide_history,
            "sky": estimate_sky(guide_history),
        })
    return {
        "rigs": rigs,
        "summary": {
            "total": len(rigs),
            "active": sum(1 for r in rigs if r["active_state"] == "green"),
            "warnings": sum(1 for r in rigs if r["guide_warning"]),
        },
    }
