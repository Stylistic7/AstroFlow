"""The Folders page's API (`web_server/backend/storage.py`): the user's input
and output folders, kept in the durable store."""
from types import SimpleNamespace

import flask
import pytest

import web_server.backend.storage as storage


@pytest.fixture
def client(monkeypatch):
    ctx = SimpleNamespace(input_dirs=None, output_dir=None)
    restarts = []
    monkeypatch.setattr(storage, "context", ctx)
    import state.actor
    monkeypatch.setattr(state.actor, "state", SimpleNamespace(drain_and_restart=lambda: restarts.append(1)))
    app = flask.Flask(__name__)
    app.register_blueprint(storage.storage_bp)
    c = app.test_client()
    c.ctx, c.restarts = ctx, restarts
    return c


def test_a_fresh_install_has_no_folders(client):
    assert client.get("/api/storage").get_json() == {"input_dirs": [], "output_dir": None}


def test_saving_folders_stores_them_and_reruns_the_graph(client, tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir(), out.mkdir()
    r = client.put("/api/storage", json={"input_dirs": [str(raw)], "output_dir": str(out)})
    assert r.status_code == 200
    assert client.ctx.input_dirs == [raw] and client.ctx.output_dir == out
    assert client.restarts == [1]
    client.put("/api/storage", json={"input_dirs": [str(raw)], "output_dir": str(out)})
    assert client.restarts == [1], "saving the same folders again changes nothing"


@pytest.mark.parametrize("body, message", [
    ({"input_dirs": [], "output_dir": "OUT"}, b"at least one input folder"),
    ({"input_dirs": ["relative/raw"], "output_dir": "OUT"}, b"full path"),
    ({"input_dirs": ["MISSING"], "output_dir": "OUT"}, b"not a folder"),
    ({"input_dirs": ["RAW", "RAW"], "output_dir": "OUT"}, b"listed twice"),
    ({"input_dirs": ["RAW"], "output_dir": "RAW/out"}, b"inside an input folder"),
])
def test_bad_folders_are_refused_and_nothing_changes(client, tmp_path, body, message):
    (tmp_path / "raw" / "out").mkdir(parents=True)
    (tmp_path / "out").mkdir()
    subst = {"RAW": str(tmp_path / "raw"), "OUT": str(tmp_path / "out"), "MISSING": str(tmp_path / "missing")}

    def fill(s):
        for k, v in subst.items():
            s = s.replace(k, v)
        return s
    body = {"input_dirs": [fill(d) for d in body["input_dirs"]], "output_dir": fill(body["output_dir"])}
    r = client.put("/api/storage", json=body)
    assert r.status_code == 400 and message in r.data
    assert client.ctx.input_dirs is None and client.restarts == []
