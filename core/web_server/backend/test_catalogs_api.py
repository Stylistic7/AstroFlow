"""The Catalogs page's API (`web_server/backend/catalogs_api.py`): the index
folder and MARS database, kept in the durable store and pushed to every Mac."""
from types import SimpleNamespace

import flask
import pytest

import web_server.backend.catalogs_api as catalogs_api
from catalogs.astrometry_index import INDEX_FILES


@pytest.fixture
def client(monkeypatch):
    ctx = SimpleNamespace(astrometry_index_dir=None, mars_database=None)
    pushes, restarts = [], []
    monkeypatch.setattr(catalogs_api, "context", ctx)
    monkeypatch.setattr(catalogs_api.node_settings, "push_to_cluster", lambda: pushes.append(1))
    import state.actor
    monkeypatch.setattr(state.actor, "state", SimpleNamespace(drain_and_restart=lambda: restarts.append(1)))
    app = flask.Flask(__name__)
    app.register_blueprint(catalogs_api.catalogs_bp)
    c = app.test_client()
    c.ctx, c.pushes, c.restarts = ctx, pushes, restarts
    return c


def test_saving_the_index_folder_creates_it_pushes_it_and_starts_the_first_run(client, tmp_path):
    index = tmp_path / "astrometry-net"
    mars = tmp_path / "MARS.xmars"
    mars.touch()
    r = client.put("/api/catalogs", json={"astrometry_index_dir": str(index), "mars_database": str(mars)})
    assert r.status_code == 200, r.data
    assert client.ctx.astrometry_index_dir == index and index.is_dir()
    assert client.ctx.mars_database == mars
    assert client.pushes == [1] and client.restarts == [1]
    client.put("/api/catalogs", json={"astrometry_index_dir": str(tmp_path / "elsewhere")})
    assert client.ctx.mars_database is None
    assert client.pushes == [1, 1] and client.restarts == [1], "only the first folder starts a run"


@pytest.mark.parametrize("body, message", [
    ({"astrometry_index_dir": "relative"}, b"full path"),
    ({"astrometry_index_dir": "MISSING/index"}, b"not a folder"),
    ({"astrometry_index_dir": "TMP", "mars_database": "TMP/none.xmars"}, b"not a file"),
])
def test_bad_choices_are_refused(client, tmp_path, body, message):
    body = {k: v.replace("TMP", str(tmp_path)).replace("MISSING", str(tmp_path / "missing"))
            for k, v in body.items()}
    r = client.put("/api/catalogs", json=body)
    assert r.status_code == 400 and message in r.data
    assert client.ctx.astrometry_index_dir is None and client.pushes == []


def test_index_status_counts_the_whole_files(client, tmp_path):
    assert client.get("/api/catalogs/index_status").get_json() is None
    client.ctx.astrometry_index_dir = tmp_path
    for name, _ in INDEX_FILES[:3]:
        (tmp_path / name).touch()
    (tmp_path / (INDEX_FILES[3][0] + ".part")).touch()
    status = client.get("/api/catalogs/index_status").get_json()
    assert status["present"] == 3 and status["total"] == len(INDEX_FILES) == 13 + 7 * 48
    assert not status["downloading"]
