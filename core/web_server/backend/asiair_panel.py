"""`GET /api/asiair/devices` — the ASIAir devices dropdown payload; see
`asiair.panel_state.compute_panel_state`."""
from __future__ import annotations

from flask import Blueprint, jsonify

from asiair.panel_state import compute_panel_state


asiair_panel_bp = Blueprint("asiair_panel", __name__)


@asiair_panel_bp.route("/api/asiair/devices")
def get_asiair_devices():
    return jsonify(compute_panel_state())
