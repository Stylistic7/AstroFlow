"""The ASIAir devices dropdown.

* `GET /api/asiair/devices` — the panel payload (`asiair.panel_state`).
* `PUT /api/asiair/devices/<key>/cooling` — the slow-ramp request for one
  rig: `{"enabled": bool, "target_c": number}`, persisted in
  `context["asiair_cooling"]` and stepped by the live poller.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from asiair.panel_state import compute_panel_state
from context import local_context as context
from web_server.backend.announcer import announce


asiair_panel_bp = Blueprint("asiair_panel", __name__)


@asiair_panel_bp.route("/api/asiair/devices")
def get_asiair_devices():
    return jsonify(compute_panel_state())


@asiair_panel_bp.route("/api/asiair/devices/<key>/cooling", methods=["PUT"])
def put_cooling(key):
    body = request.get_json(silent=True) or {}
    try:
        target_c = float(body["target_c"])
        enabled = bool(body["enabled"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "body needs enabled (bool) and target_c (number)"}), 400
    if not -40.0 <= target_c <= 30.0:
        return jsonify({"error": "target_c must be within -40..30"}), 400
    current = dict(context.get("asiair_cooling", None) or {})
    current[key] = {"enabled": enabled, "target_c": target_c}
    context["asiair_cooling"] = current
    announce("asiair_devices")
    return jsonify({"key": key, **current[key]})
