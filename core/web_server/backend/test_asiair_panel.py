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


def test_put_cooling_persists_the_request_and_announces():
    import web_server.backend.asiair_panel as bp

    class _Store(dict):
        def get(self, key, default=None):
            return super().get(key, default)
    store, announced = _Store(), []
    saved = bp.context, bp.announce
    bp.context, bp.announce = store, lambda event, data=None: announced.append(event)
    try:
        app = flask.Flask(__name__); app.register_blueprint(asiair_panel_bp)
        c = app.test_client()
        assert c.put("/api/asiair/devices/96578d3e/cooling", json={"enabled": True, "target_c": -10}).status_code == 200
        assert store["asiair_cooling"] == {"96578d3e": {"enabled": True, "target_c": -10.0}} and announced == ["asiair_devices"]
        assert c.put("/api/asiair/devices/96578d3e/cooling", json={"enabled": True, "target_c": -60}).status_code == 400
        assert c.put("/api/asiair/devices/96578d3e/cooling", json={"enabled": "yes"}).status_code == 400
    finally:
        bp.context, bp.announce = saved


if __name__ == "__main__":
    test_get_devices_serves_the_rig_rows()
    print("ok test_get_devices_serves_the_rig_rows")
    test_put_cooling_persists_the_request_and_announces()
    print("ok test_put_cooling_persists_the_request_and_announces")
