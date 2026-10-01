"""`/thumbnail/stack/<id>` is the target page's card image and says which
image it is: the master's default finish (status live / out_of_date,
X-Producer master), else the best raw subframe (status absent, X-Producer
subframe), else 404 (2026-10-01)."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def client(test_context):
    from web_server.backend.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _fake_stack(monkeypatch):
    from web_server.backend import thumbnails as th
    stack = SimpleNamespace(stable_hash="s1")
    monkeypatch.setattr(th, "context", SimpleNamespace(stacks={"s1": SimpleNamespace(get=lambda: stack)}))
    return stack


def test_master_case_serves_the_default_finish_with_its_status(client, monkeypatch):
    import catalog
    import finishes.preview as preview
    _fake_stack(monkeypatch)
    bound = SimpleNamespace(object_type="stack", object_id="s1", pipeline="P", name="H")
    monkeypatch.setattr(catalog, "default_finish_for", lambda t, i: bound)
    monkeypatch.setattr(preview, "render_finish_preview", lambda *a: (b"PNGmaster", "out_of_date"))
    r = client.get("/thumbnail/stack/s1")
    assert r.status_code == 200 and r.data == b"PNGmaster"
    assert r.headers["X-Producer"] == "master"
    assert r.headers["X-Product-Status"] == "out_of_date"


def test_unstacked_case_serves_the_best_subframe_as_absent(client, monkeypatch):
    import catalog
    from web_server.backend import thumbnails as th
    _fake_stack(monkeypatch)
    monkeypatch.setattr(catalog, "default_finish_for", lambda t, i: None)
    monkeypatch.setattr(th, "_stack_subframe_thumbnail", lambda entity, args: b"PNGsub")
    r = client.get("/thumbnail/stack/s1")
    assert r.status_code == 200 and r.data == b"PNGsub"
    assert r.headers["X-Producer"] == "subframe"
    assert r.headers["X-Product-Status"] == "absent"


def test_nothing_readable_is_404(client, monkeypatch):
    import catalog
    from web_server.backend import thumbnails as th
    _fake_stack(monkeypatch)
    monkeypatch.setattr(catalog, "default_finish_for", lambda t, i: None)
    monkeypatch.setattr(th, "_stack_subframe_thumbnail", lambda entity, args: None)
    assert client.get("/thumbnail/stack/s1").status_code == 404
