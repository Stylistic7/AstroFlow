"""The canonical-preview status contract (2026-08-31).

The endpoint computes freshness and must never drop it again: the status
travels in `X-Product-Status`, and `?fresh=1` refuses to serve last-good.
`render_canonical_preview` is monkeypatched — these are route-contract
tests, not finish-rendering tests."""
import flask
import pytest

import web_server.backend.thumbnails as th


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.register_blueprint(th.thumbnails_bp)
    return app.test_client()


def _patch_render(monkeypatch, result):
    import finishes.preview as fp
    monkeypatch.setattr(fp, "render_canonical_preview",
                        lambda *a, **k: result)


def test_status_header_live(client, monkeypatch):
    _patch_render(monkeypatch, (b"png-bytes", "live"))
    r = client.get("/thumbnail/preview/mosaic/abc123")
    assert r.status_code == 200
    assert r.headers["X-Product-Status"] == "live"
    assert r.data == b"png-bytes"


def test_status_header_out_of_date(client, monkeypatch):
    _patch_render(monkeypatch, (b"old-png", "out_of_date"))
    r = client.get("/thumbnail/preview/mosaic/abc123")
    assert r.status_code == 200
    assert r.headers["X-Product-Status"] == "out_of_date"


def test_fresh_refuses_stale(client, monkeypatch):
    _patch_render(monkeypatch, (b"old-png", "out_of_date"))
    r = client.get("/thumbnail/preview/mosaic/abc123?fresh=1")
    assert r.status_code == 404


def test_fresh_serves_live(client, monkeypatch):
    _patch_render(monkeypatch, (b"png", "live"))
    r = client.get("/thumbnail/preview/mosaic/abc123?fresh=1")
    assert r.status_code == 200


def test_absent_404(client, monkeypatch):
    _patch_render(monkeypatch, None)
    r = client.get("/thumbnail/preview/mosaic/abc123")
    assert r.status_code == 404
