"""The finish-thumbnail status contract (2026-08-31, generalised 2026-09-16).

Both thumbnail routes compute freshness and must never drop it: the status
travels in `X-Product-Status`, and `?fresh=1` refuses to serve last-good.
`render_finish_preview` and `default_finish_for` are monkeypatched — these
are route-contract tests, not finish-rendering tests."""
import flask
import pytest

import web_server.backend.thumbnails as th


@pytest.fixture
def client(monkeypatch):
    app = flask.Flask(__name__)
    app.register_blueprint(th.thumbnails_bp)
    return app.test_client()


class _Bound:
    object_type, object_id, pipeline, name = "mosaic", "abc123", "P", "Hubble"


def _patch(monkeypatch, result, default=_Bound()):
    import finishes.preview as fp
    import catalog
    monkeypatch.setattr(fp, "render_finish_preview", lambda *a, **k: result)
    monkeypatch.setattr(catalog, "default_finish_for", lambda *a, **k: default)


@pytest.mark.parametrize("url", ["/thumbnail/preview/mosaic/abc123",
                                 "/tap/thumbnail/mosaic/abc123/P/Hubble"])
def test_status_header_live(client, monkeypatch, url):
    _patch(monkeypatch, (b"png-bytes", "live"))
    r = client.get(url)
    assert r.status_code == 200
    assert r.headers["X-Product-Status"] == "live"
    assert r.data == b"png-bytes"


def test_status_header_out_of_date(client, monkeypatch):
    _patch(monkeypatch, (b"old-png", "out_of_date"))
    r = client.get("/tap/thumbnail/mosaic/abc123/P/Hubble")
    assert r.status_code == 200
    assert r.headers["X-Product-Status"] == "out_of_date"


def test_fresh_refuses_stale(client, monkeypatch):
    _patch(monkeypatch, (b"old-png", "out_of_date"))
    assert client.get("/thumbnail/preview/mosaic/abc123?fresh=1").status_code == 404
    assert client.get("/tap/thumbnail/mosaic/abc123/P/Hubble?fresh=1").status_code == 404


def test_fresh_serves_live(client, monkeypatch):
    _patch(monkeypatch, (b"png", "live"))
    assert client.get("/thumbnail/preview/mosaic/abc123?fresh=1").status_code == 200


def test_absent_404(client, monkeypatch):
    _patch(monkeypatch, None)
    assert client.get("/tap/thumbnail/mosaic/abc123/P/Hubble").status_code == 404
    _patch(monkeypatch, (b"png", "live"), default=None)
    assert client.get("/thumbnail/preview/mosaic/abc123").status_code == 404
