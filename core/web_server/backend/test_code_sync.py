"""The `/api/cluster-version` endpoint: serves the version computed once per
process from the live checkout. The snapshot-serving endpoints are gone —
code moves via git; workers fence on this version and pull with `git pull`.
"""
import flask
import pytest

import web_server.backend.code_sync as cs

VERSION = "a" * 40


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(cs, "_version_cache", VERSION)
    app = flask.Flask(__name__)
    app.register_blueprint(cs.code_sync_bp)
    return app.test_client()


def test_cluster_version_serves_the_cached_version(client):
    r = client.get("/api/cluster-version")
    assert r.status_code == 200
    assert r.get_json() == {"cluster_version": VERSION}


def test_version_computes_once_then_sticks(monkeypatch, repo=None):
    calls = []
    monkeypatch.setattr(cs, "_version_cache", None)
    monkeypatch.setattr(cs, "head_cluster_version",
                        lambda repo, unc: calls.append(1) or VERSION)
    monkeypatch.setattr(cs, "_uncommitted_mode", lambda: False)
    assert cs._version() == VERSION
    assert cs._version() == VERSION
    assert len(calls) == 1, "computed once per process; deploy = restart"


def test_version_empty_when_not_a_repo(monkeypatch):
    def boom(repo, unc):
        raise RuntimeError("not a git repo")
    monkeypatch.setattr(cs, "_version_cache", None)
    monkeypatch.setattr(cs, "head_cluster_version", boom)
    monkeypatch.setattr(cs, "_uncommitted_mode", lambda: False)
    assert cs._version() == ""   # advertises nothing -> workers won't join


def test_snapshot_endpoint_is_gone(client):
    assert client.get("/api/code-snapshot").status_code == 404
