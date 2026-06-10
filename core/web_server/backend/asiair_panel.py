"""Flask blueprint for the ASIAir devices dropdown panel.

Two routes:

* `GET /api/asiair/devices` — payload for the panel; see
  `asiair.panel_state.compute_panel_state`.
* `PATCH /api/asiair/devices/<camera_hash>/nickname` — set / update the
  nickname for one camera, keyed by `camera_id_hash`. Rejects
  zero-length camera_hash (the "pending:" rows from
  `compute_panel_state`) with 400 — there's no stable identity to bind
  the nickname to yet.

Nicknames are persisted in `context["asiair_device_nicknames"]`
(`dict[camera_hash, nickname]`). They never get written into FITS /
XISF / sidecar files — explicit "AstroFlow never modifies user data"
rule, mirrored from the plan.
"""
from __future__ import annotations

from flask import Blueprint, request, jsonify

from context import local_context as context
from web_server.backend.announcer import announce
from asiair.panel_state import compute_panel_state


asiair_panel_bp = Blueprint("asiair_panel", __name__)


@asiair_panel_bp.route("/api/asiair/devices")
def get_asiair_devices():
    return jsonify(compute_panel_state())


@asiair_panel_bp.route(
    "/api/asiair/devices/<camera_hash>/nickname", methods=["PATCH"],
)
def patch_asiair_nickname(camera_hash):
    if not camera_hash:
        # Pending rows have an empty `camera_hash_short` and a synthetic
        # `pending:<hostname>` key — no stable id to bind to. The UI
        # disables the rename input for these rows, but reject defensively.
        return jsonify({
            "error": "Cannot nickname a rig before its fingerprint anchors.",
        }), 400

    body = request.get_json(silent=True) or {}
    nickname = body.get("nickname")
    if nickname is not None and not isinstance(nickname, str):
        return jsonify({"error": "nickname must be a string or null"}), 400
    # Trim — empty string means "clear the nickname" so we delete the
    # entry, keeping the dict free of zero-value rows.
    nickname = nickname.strip() if isinstance(nickname, str) else None

    current = dict(context.get("asiair_device_nicknames", None) or {})
    if not nickname:
        current.pop(camera_hash, None)
    else:
        current[camera_hash] = nickname
    context["asiair_device_nicknames"] = current

    announce("asiair_devices")
    return jsonify({
        "camera_hash": camera_hash,
        "nickname": nickname or None,
    })
