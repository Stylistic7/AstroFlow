"""A request that needs Ray before the head's Ray is up is told to retry
(503 + Retry-After), not handed a 500 (`web_server/backend/app.py`)."""
import web_server.backend.tools_api as tools_api
from ray_infra.ray_init import RayNotUpError
from web_server.backend.app import app


def test_a_request_before_ray_is_up_is_told_to_retry(monkeypatch):
    def not_up():
        raise RayNotUpError("Could not find any running Ray instance.")
    monkeypatch.setattr(tools_api.node_settings, "current", not_up)
    r = app.test_client().get("/api/node_settings")
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "5"
    assert r.get_json()["status"] == "starting"
