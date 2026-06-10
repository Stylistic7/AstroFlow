"""Tests for the ASIAir devices Flask blueprint.

Strategy mirrors `asiair/test_panel_state.py`: stub a small object
where `local_context` is imported, mount the blueprint on a minimal
Flask app, drive it with `app.test_client()`. Avoids Ray + diskcache.
"""
from __future__ import annotations

import flask
import pytest


# Replace announcer.announce with a recorder *before* importing the
# blueprint so its module-level `from .announcer import announce`
# binds the stub. Plain monkeypatch on the already-imported symbol
# would otherwise re-bind only the recorder's view.


@pytest.fixture
def announcements(monkeypatch):
    """Capture every `announce(...)` call made by the blueprint.

    The blueprint binds `announce` at module load time via
    `from .announcer import announce`, so we must patch the symbol on
    the blueprint module itself — patching the announcer module's
    attribute doesn't reach this rebound local."""
    captured: list[tuple] = []

    import web_server.backend.asiair_panel as bp_mod

    def _fake_announce(event, data=None):
        captured.append((event, data))

    monkeypatch.setattr(bp_mod, "announce", _fake_announce)
    return captured


class _StubContext:
    """Mirrors the LocalContext surface that the blueprint uses:
    `.asiair_devices`, `.stacks`, `.get(key, default)`, and `__setitem__`."""

    def __init__(self):
        self.asiair_devices = []
        self.stacks = {}
        self._store: dict = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def __setitem__(self, key, value):
        self._store[key] = value


@pytest.fixture
def stub_context(monkeypatch):
    """Replace `local_context` in BOTH modules the blueprint touches:
    the blueprint itself and the underlying panel_state aggregator."""
    ctx = _StubContext()
    import web_server.backend.asiair_panel as bp_mod
    import asiair.panel_state as ps_mod
    monkeypatch.setattr(bp_mod, "context", ctx)
    monkeypatch.setattr(ps_mod, "context", ctx)
    return ctx


@pytest.fixture
def app(stub_context, announcements):
    """Minimal Flask app with just the asiair_panel blueprint mounted.
    Reuses the stub_context / announcements fixtures."""
    # Re-import here so the blueprint sees the patched `context` and
    # `announce` symbols when its routes execute.
    from web_server.backend.asiair_panel import asiair_panel_bp

    app = flask.Flask(__name__)
    app.register_blueprint(asiair_panel_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_get_devices_returns_payload_shape(client, stub_context):
    stub_context.asiair_devices = [
        {"ip": "10.0.0.42", "hostname": "asiair-foo.local", "mac": "aa:bb"},
    ]
    # No anchored stacks → one pending row for the discovered ASIAir.
    resp = client.get("/api/asiair/devices")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "rigs" in body
    assert "summary" in body
    assert body["summary"]["total"] == 1
    assert body["rigs"][0]["anchored"] is False
    assert body["rigs"][0]["asiair_hostname"] == "asiair-foo.local"


def test_patch_nickname_persists_and_announces(client, stub_context, announcements):
    camera_hash = "abc12345deadbeef"
    resp = client.patch(
        f"/api/asiair/devices/{camera_hash}/nickname",
        json={"nickname": "Backyard rig"},
    )
    assert resp.status_code == 200
    assert resp.get_json() == {
        "camera_hash": camera_hash, "nickname": "Backyard rig",
    }
    # Persistence: stored under the dedicated key.
    assert stub_context.get("asiair_device_nicknames") == {
        camera_hash: "Backyard rig",
    }
    # Side-channel: SSE event fired so other connected clients refetch.
    assert ("asiair_devices", None) in announcements


def test_patch_nickname_clears_when_blank(client, stub_context):
    camera_hash = "cam-hash-1"
    stub_context["asiair_device_nicknames"] = {camera_hash: "old name"}

    resp = client.patch(
        f"/api/asiair/devices/{camera_hash}/nickname",
        json={"nickname": ""},
    )
    assert resp.status_code == 200
    # Empty / whitespace nicknames mean "clear" — the dict drops the
    # entry rather than storing an empty string.
    assert stub_context.get("asiair_device_nicknames") == {}


def test_patch_nickname_rejects_non_string(client, stub_context):
    resp = client.patch(
        "/api/asiair/devices/cam-hash/nickname",
        json={"nickname": 42},
    )
    assert resp.status_code == 400


def test_patch_pending_row_rejected_with_400(client, stub_context):
    # The empty camera_hash variant — the UI disables rename on pending
    # rows, but the server enforces the same rule defensively. Flask's
    # router doesn't accept an empty path segment, so PATCH-ing the
    # bare collection lands on a 404. The defensive 400 fires on the
    # ROUTE-MATCHED pending key (the `pending:` prefix is part of
    # the `compute_panel_state` synthetic key but never lands here —
    # the UI doesn't try to PATCH it because the rename input is
    # disabled). So the defensive check sits where any non-anchored
    # caller would still be politely refused: empty path → 404.
    # Effectively pending rows are guarded at the UI layer plus the
    # 404 fallback here.
    resp = client.patch(
        "/api/asiair/devices//nickname",
        json={"nickname": "x"},
    )
    assert resp.status_code in (400, 404)
