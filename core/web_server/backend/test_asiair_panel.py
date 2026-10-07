"""The ASIAir devices route serves `compute_panel_state`. No Ray: the
aggregator's context and guide history are stubbed as in
`asiair/test_panel_state.py`."""
from __future__ import annotations

import flask

from asiair.test_panel_state import rig, rigs_in_context
from web_server.backend.asiair_panel import asiair_panel_bp


def test_get_devices_serves_the_rig_rows():
    app = flask.Flask(__name__)
    app.register_blueprint(asiair_panel_bp)
    with rigs_in_context([rig(reachable=True, capture_state="idle")]):
        resp = app.test_client().get("/api/asiair/devices")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["summary"] == {"total": 1, "active": 0, "warnings": 0}
    (row,) = body["rigs"]
    assert (row["label"], row["sn"], row["ip"], row["active_state"]) == ("ASIAIR_RC71 · 351 mm", "96578d3e", "10.0.0.28", "yellow")


if __name__ == "__main__":
    test_get_devices_serves_the_rig_rows()
    print("ok test_get_devices_serves_the_rig_rows")
