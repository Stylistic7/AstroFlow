"""Head-side pull-deploy endpoints. They serve the head's FROZEN CHECKOUT
(no git at request time), so the test builds a checkout dir with a recorded
version and points the blueprint at it. No real cache or cluster.
"""
import io
import tarfile

import flask
import pytest

import web_server.backend.code_sync as cs
from state.code_version import SYNCED_SHA_FILE

VERSION = "a" * 40


@pytest.fixture
def client(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "core").mkdir(parents=True)
    (checkout / "core" / "main.py").write_text("print('v1')\n")
    (checkout / SYNCED_SHA_FILE).write_text(VERSION)
    monkeypatch.setattr(cs, "checkout_dir", lambda: checkout)

    app = flask.Flask(__name__)
    app.register_blueprint(cs.code_sync_bp)
    c = app.test_client()
    c._checkout = checkout
    return c


def test_cluster_version_is_the_frozen_version(client):
    r = client.get("/api/cluster-version")
    assert r.status_code == 200
    assert r.get_json()["cluster_version"] == VERSION


def test_snapshot_serves_frozen_checkout_with_header(client):
    r = client.get(f"/api/code-snapshot?sha={VERSION}")
    assert r.status_code == 200
    assert r.headers["X-AstroFlow-SHA"] == VERSION
    with tarfile.open(fileobj=io.BytesIO(r.data), mode="r:gz") as tf:
        names = tf.getnames()
    assert "core/main.py" in names
    assert SYNCED_SHA_FILE not in names  # sidecar isn't shipped


def test_snapshot_rejects_mismatched_version(client):
    # Head has redeployed since the worker saw the old version → 404, re-discover.
    assert client.get("/api/code-snapshot?sha=" + "b" * 40).status_code == 404


def test_snapshot_without_sha_serves_current(client):
    assert client.get("/api/code-snapshot").status_code == 200


def test_endpoints_when_nothing_frozen_yet(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(cs, "checkout_dir", lambda: empty)
    app = flask.Flask(__name__)
    app.register_blueprint(cs.code_sync_bp)
    c = app.test_client()
    assert c.get("/api/cluster-version").get_json()["cluster_version"] == ""
    assert c.get("/api/code-snapshot").status_code == 503
