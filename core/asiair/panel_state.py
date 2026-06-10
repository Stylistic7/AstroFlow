"""Aggregator for the ASIAir devices dropdown panel.

Joins three live sources to produce one row per known camera plus
pending rows for ASIAirs whose camera identity hasn't anchored yet:

* `context.asiair_devices` — Zeroconf-discovered ASIAirs (live).
* `context.stacks` — `Stack` objects each carrying `camera_id_hash`
  (derived from defect-pixel consensus). One unique `camera_id_hash`
  = one camera.
* `context["asiair_device_nicknames"]` — user-assigned nicknames,
  keyed by `camera_id_hash`. Never written into FITS / sidecar /
  XISF — explicit "AstroFlow never modifies user data" rule.

There is no live ASIAir API (no HTTP/WebSocket for cooler temp,
target name, mount state) — everything below derives from the
already-downloaded files' metadata as carried on the Stack objects.

The panel is keyed by CAMERA, not by ASIAir, because the user
established the rigs are fungible: "I only care about the main
camera's identity". The ASIAir hostname is shown per-row as
informational context only.
"""
from __future__ import annotations

from typing import Any

from context import local_context as context


_ACTIVE_GREEN_SECONDS = 30 * 60        # < 30 min
_ACTIVE_YELLOW_SECONDS = 24 * 60 * 60  # < 24 h


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
    nicknames: dict[str, str] = (
        context.get("asiair_device_nicknames", None) or {}
    )

    # Group stacks by `camera_id_hash`. Unanchored stacks (None hash)
    # are skipped — they show up as the "Awaiting fingerprint" pending
    # rows below, by virtue of an ASIAir being discovered but no row
    # claiming its camera.
    by_camera: dict[str, list[Any]] = {}
    for s_or_ref in stacks_dict.values():
        stack = _resolve_stack(s_or_ref)
        camera_hash = getattr(stack, "camera_id_hash", None)
        if camera_hash is None:
            continue
        by_camera.setdefault(camera_hash, []).append(stack)

    # Best-effort "via" attribution: there's no per-file ASIAir tag in
    # the download path today (the rig field is hardcoded "foo"; all
    # files land in `input_dirs[0]/{date}/`), so we can't tell which
    # ASIAir delivered which Stack. v1 shows the first discovered
    # ASIAir's hostname/IP on every anchored row; if multiple are
    # discovered, the user infers the rest from row order.
    primary_asiair = asiairs[0] if asiairs else None
    primary_hostname = primary_asiair["hostname"] if primary_asiair else ""
    primary_ip = primary_asiair["ip"] if primary_asiair else ""

    rigs: list[dict[str, Any]] = []
    for camera_hash, group in by_camera.items():
        # Pick the most-recently-active stack to source the row's
        # live fields (last filter, last target, last instrument).
        # `newest_date` is the maximum frame DATE-OBS across the
        # Stack — built once at identify time, no FS walk.
        group_sorted = sorted(
            group,
            key=lambda s: getattr(s, "newest_date", None) or 0,
            reverse=True,
        )
        primary_stack = group_sorted[0]
        newest_date = getattr(primary_stack, "newest_date", None)
        last_image_mtime = (
            newest_date.timestamp() if newest_date is not None else 0.0
        )

        rigs.append({
            "key": camera_hash,
            "anchored": True,
            "camera_hash_short": camera_hash[:8],
            "nickname": nicknames.get(camera_hash),
            "asiair_hostname": primary_hostname,
            "asiair_ip": primary_ip,
            "last_image_mtime": last_image_mtime,
            "last_filter": getattr(primary_stack, "filter", None),
            # Stack doesn't carry FITS OBJECT directly. We'd need a
            # frame-header lookup, which is a per-row I/O hit; defer.
            "last_object": None,
            "last_instrume": getattr(primary_stack, "camera", None),
            "active_state": _band(last_image_mtime, now_epoch),
            "frames_until_anchor": None,
        })

    # Pending rows: discovered ASIAirs in excess of known cameras get
    # a placeholder row each. The user sees "Awaiting fingerprint" and
    # the rename input stays disabled until the fingerprint anchors and
    # the row promotes to one of the camera-keyed rows above.
    n_excess = max(0, len(asiairs) - len(by_camera))
    for i in range(n_excess):
        dev = asiairs[len(by_camera) + i]
        host = dev.get("hostname") or dev.get("ip", "")
        rigs.append({
            "key": f"pending:{host}",
            "anchored": False,
            "camera_hash_short": "",
            "nickname": None,
            "asiair_hostname": dev.get("hostname", ""),
            "asiair_ip": dev.get("ip", ""),
            "last_image_mtime": 0.0,
            "last_filter": None,
            "last_object": None,
            "last_instrume": None,
            "active_state": "gray",
            "frames_until_anchor": None,
        })

    active = sum(1 for r in rigs if r["active_state"] == "green")
    return {
        "rigs": rigs,
        "summary": {"total": len(rigs), "active": active},
    }
