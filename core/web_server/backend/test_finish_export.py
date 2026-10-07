"""The Open-in-Photoshop routes (2026-10-04): validate, resolve the view,
launch the bake as its own process, and report its status."""
import json
from types import SimpleNamespace as NS

import pytest

from web_server.backend import finish_export as fe

URL = "/api/finish_export/target/abc/Hybrid%3A%20P/Hubble%20Pinned%20GHS"


@pytest.fixture
def client(tmp_path, monkeypatch):
    from web_server.backend.app import app
    import catalog
    import finishes.export as ex
    monkeypatch.setattr(fe, "EXPORT_DIR", tmp_path)
    monkeypatch.setattr(fe, "_JOBS", tmp_path / ".jobs")
    monkeypatch.setattr(catalog, "resolve_finish", lambda *a: NS(inputs=[
        NS(tap="Linear SHO", path="/data/nb.tiff"), NS(tap="Linear RGB Stars", path=None)]))
    monkeypatch.setattr(ex, "newest_photoshop", lambda: "/Applications/PS.app")
    launched = []
    monkeypatch.setattr(fe.subprocess, "Popen", lambda cmd, **k: launched.append(cmd))
    with app.test_client() as c:
        yield c, launched


def test_start_writes_a_job_and_launches_the_bake(client):
    c, launched = client
    r = c.post(URL, json={"params": {"strS": 6.0}})
    assert r.status_code == 200
    job = r.get_json()["job"]
    assert launched and launched[0][1:3] == ["-m", "finishes.export"]
    spec = json.loads(open(launched[0][3]).read())
    assert spec["primary"] == "Linear SHO" and spec["paths"] == ["/data/nb.tiff", None]
    assert spec["params"] == {"strS": 6.0} and spec["app"] == "/Applications/PS.app"
    assert spec["dest"].endswith(".tif") and "Hubble-Pinned-GHS-target-abc-" in spec["dest"]
    assert c.get(f"/api/finish_export/job/{job}").get_json()["state"] == "running"


@pytest.mark.parametrize("body", [{}, {"params": {}}, {"params": {"a": "x"}}])
def test_bad_params_are_rejected(client, body):
    c, _ = client
    assert c.post(URL, json=body).status_code == 400


def test_unknown_job_is_404(client):
    c, _ = client
    assert c.get("/api/finish_export/job/000000000000").status_code == 404
    assert c.get("/api/finish_export/job/../x").status_code == 404
