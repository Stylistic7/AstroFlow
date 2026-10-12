"""`_store_measured_viewbox` files the call hash of the cart item's OWN
measured viewbox — the one spanning the most channels — so the web's polygon
overlay projects onto the canvas the image is placed on (Ray-free).

The module's `context` is REPLACED with a recorder, never patched attribute by
attribute: `Context.__setattr__` writes the live store."""
import compute_graph.compute_graph as cg


class _Node:
    def __init__(self, name, args, content_hash):
        def task():
            pass
        task.__name__ = name
        self.task, self.args, self.content_hash = task, args, content_hash


class _Recorder:
    def __init__(self):
        self.written = {}

    def set(self, key, value):
        self.written[key] = value


def test_files_the_viewbox_spanning_the_most_channels(monkeypatch):
    panel = _Node("compute_viewbox_measured", ("Master", True, ("a",), "c1"), "panelvb")
    mosaic = _Node("compute_viewbox_measured", ("Master", False, ("a", "b"), "c1", "c2"), "mosaicvb")
    place = _Node("place_image", ("x",), "place")
    rec = _Recorder()
    monkeypatch.setattr(cg, "context", rec)
    cg._store_measured_viewbox(("src", "stable", "content", "pipe"), {place: [mosaic], mosaic: [panel]})
    assert rec.written == {("measured_viewbox", "src", "stable", "content", "pipe"): "mosaicvb"}


def test_no_measured_viewbox_files_nothing(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(cg, "context", rec)
    cg._store_measured_viewbox(("k",), {_Node("place_image", (), "p"): []})
    assert rec.written == {}
