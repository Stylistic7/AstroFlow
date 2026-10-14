"""Saved finish looks (2026-10-04): save, list, restore-by-data and delete per
(object, pipeline, finish), stored durably, with a server-rendered thumbnail."""
import cv2
import numpy as np
import pytest

from context.cache import is_durable_key
from web_server.backend import finish_saves

URL = "/api/finish_saves/target/abc/Hybrid%3A%20P/Hubble"


class _Store(dict):
    def get(self, k, default=None):
        return super().get(k, default)

    def set(self, k, v):
        self[k] = v


@pytest.fixture
def client(monkeypatch):
    from web_server.backend.app import app
    store = _Store()
    monkeypatch.setattr(finish_saves, "context", store)
    monkeypatch.setattr(finish_saves, "_thumbnail", lambda *a: "data:image/jpeg;base64,AA==")
    with app.test_client() as c:
        yield c, store


def test_save_list_and_delete(client):
    c, store = client
    assert c.get(URL).get_json() == {"saves": []}
    r = c.post(URL, json={"params": {"strS": 6.15, "levelH": 0.04, "show_clipping": 0}})
    saves = r.get_json()["saves"]
    assert len(saves) == 1 and saves[0]["params"] == {"strS": 6.15, "levelH": 0.04, "show_clipping": 0.0}
    assert saves[0]["thumbnail"].startswith("data:image/jpeg")
    c.post(URL, json={"params": {"strS": 7.0}})
    listed = c.get(URL).get_json()["saves"]
    assert [s["params"]["strS"] for s in listed] == [6.15, 7.0]
    assert ("finish_saves", "target", "abc", "Hybrid: P", "Hubble") in store
    left = c.delete(f"{URL}/{listed[0]['id']}").get_json()["saves"]
    assert [s["params"]["strS"] for s in left] == [7.0]


@pytest.mark.parametrize("body", [{}, {"params": {}}, {"params": {"a": "x"}},
                                  {"params": {"a": True}}, {"params": [1]}])
def test_bad_params_are_rejected(client, body):
    c, _ = client
    assert c.post(URL, json=body).status_code == 400


def test_saves_live_in_the_durable_store():
    assert is_durable_key(("finish_saves", "target", "abc", "P", "F"))


def test_thumbnail_is_a_small_jpeg(monkeypatch):
    import finishes.preview as pv
    img = np.full((300, 600, 3), 128, np.uint8)
    png = cv2.imencode(".png", img)[1].tobytes()
    seen = {}

    def fake(object_type, object_id, pipeline, name, min_px, values):
        seen.update(values=values, min_px=min_px)
        return png, "live"

    monkeypatch.setattr(pv, "render_finish_preview", fake)
    url = finish_saves._thumbnail("target", "abc", "P", "F", {"strS": 6.0})
    assert url.startswith("data:image/jpeg;base64,") and seen["values"] == {"strS": 6.0}
    import base64
    jpg = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert jpg.shape[1] == 160 and jpg.shape[0] == 80
