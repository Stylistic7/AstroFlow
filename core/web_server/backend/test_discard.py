"""Unit tests for the discard-results feature (Ray-free, no fixtures).

Covers the `_discard` helpers, the two Flask routes (mounted on a minimal app
and driven with `test_client()`, mirroring `test_asiair_panel.py`), and
`trace_paths.invalidate_tap_snapshot`. Each test stands alone (tempfiles +
unittest.mock), so it runs under pytest AND can be imported + called directly
when the xdist/Ray harness is unavailable.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import flask

from web_server.backend import _discard


def _make_atom(d: Path, op: str, stem: str) -> Path:
    """Create a fake op-output atom (primary .fits + .json sidecar + .tiff
    pyramid + .context) under d/op/, return the primary path."""
    opdir = d / op
    opdir.mkdir(parents=True, exist_ok=True)
    primary = opdir / f"{stem}.fits"
    primary.write_bytes(b"x" * 10)
    _discard.sidecar_path_for(primary).write_text("{}")
    tiff = primary.with_suffix(".tiff")
    tiff.write_bytes(b"y" * 20)
    tiff.with_suffix(".context").write_bytes(b"z" * 5)
    return primary


def _fake_entries(d: Path) -> list[dict]:
    return [
        {"op_name": "post_process_stack_oss",
         "path": str(_make_atom(d, "post_process_stack_oss", "aaaa")),
         "display": "Post-Processed", "is_final": False},
        {"op_name": "post_process_stack_oss",
         "path": str(_make_atom(d, "post_process_stack_oss", "bbbb")),
         "display": "Post-Processed", "is_final": False},
        {"op_name": "merge_images_oss",
         "path": str(_make_atom(d, "merge_images_oss", "cccc")),
         "display": "Mosaic", "is_final": True},
    ]


def test_group_ops_buckets_by_op_name():
    with tempfile.TemporaryDirectory() as td:
        entries = _fake_entries(Path(td))
        with patch.object(_discard, "traced_paths_for", return_value=entries):
            ops = _discard.group_ops(("mosaic", "x", "P"))
        by = {g["op_name"]: g for g in ops}
        assert by["post_process_stack_oss"]["count"] == 2
        assert by["post_process_stack_oss"]["on_disk"] == 2
        assert by["merge_images_oss"]["count"] == 1
        assert by["merge_images_oss"]["is_final"] is True
        # bytes = primary(10) + tiff(20) per existing node
        assert by["post_process_stack_oss"]["bytes"] == 2 * 30
        assert by["merge_images_oss"]["bytes"] == 30


def test_discard_ops_deletes_selected_atoms_and_cache():
    with tempfile.TemporaryDirectory() as td:
        entries = _fake_entries(Path(td))
        cache = dict(
            ((e["op_name"], Path(e["path"]).stem), "v") for e in entries
        )
        with patch.object(_discard, "traced_paths_for", return_value=entries), \
             patch.object(_discard, "get_cache", return_value=cache):
            res = _discard.discard_ops(
                ("mosaic", "x", "P"), {"post_process_stack_oss"}
            )
        # 2 selected nodes × 4 atom files each
        assert res["files_deleted"] == 8
        assert res["cache_deleted"] == 2
        for e in entries:
            primary = Path(e["path"])
            key = (e["op_name"], primary.stem)
            if e["op_name"] == "post_process_stack_oss":
                assert not primary.exists()              # selected → gone
                assert key not in cache
            else:
                assert primary.exists()                  # unselected → untouched
                assert key in cache


def test_discard_ops_empty_selection_is_noop():
    with tempfile.TemporaryDirectory() as td:
        entries = _fake_entries(Path(td))
        with patch.object(_discard, "traced_paths_for", return_value=entries), \
             patch.object(_discard, "get_cache", return_value={}):
            res = _discard.discard_ops(("mosaic", "x", "P"), set())
        assert res == {"files_deleted": 0, "cache_deleted": 0}
        assert all(Path(e["path"]).exists() for e in entries)


# --- Flask routes (mounted on a minimal app, mirror test_asiair_panel.py) ---

def _client():
    from web_server.backend.metadata import metadata_bp
    app = flask.Flask(__name__)
    app.register_blueprint(metadata_bp)
    return app.test_client()


def test_pipeline_results_route_groups_ops():
    from web_server.backend import metadata
    fake = [{"op_name": "merge_images_oss", "display": "Mosaic",
             "is_final": True, "count": 6, "on_disk": 6, "bytes": 100}]
    with patch.object(metadata, "_resolve_entity", return_value=object()), \
         patch.object(_discard, "group_ops", return_value=fake):
        r = _client().get("/api/pipeline_results/mosaic/abc/Hybrid%3A%20X")
    assert r.status_code == 200
    body = r.get_json()
    assert body["pipeline"] == "Hybrid: X"      # %3A %20 decoded by the router
    assert body["ops"] == fake


def test_pipeline_results_route_404_when_entity_missing():
    from web_server.backend import metadata
    with patch.object(metadata, "_resolve_entity", return_value=None):
        r = _client().get("/api/pipeline_results/mosaic/missing/P")
    assert r.status_code == 404


def test_discard_results_route_discards_and_reruns():
    from web_server.backend import metadata
    import state.actor
    import web_server.backend.announcer as announcer
    fake_state = MagicMock()
    announced = []
    with patch.object(metadata, "_resolve_entity", return_value=object()), \
         patch.object(_discard, "discard_ops",
                      return_value={"files_deleted": 8, "cache_deleted": 2}) as disc, \
         patch("dependency_graph.trace_paths.invalidate_tap_snapshot",
               return_value=True) as inval, \
         patch.object(state.actor, "state", fake_state), \
         patch.object(announcer, "announce",
                      side_effect=lambda *a, **k: announced.append(a)):
        r = _client().post(
            "/api/discard_results/mosaic/abc/Hybrid%3A%20X",
            json={"op_names": ["merge_images_oss", "post_process_stack_oss"]},
        )
    assert r.status_code == 200
    body = r.get_json()
    assert body["files_deleted"] == 8 and body["cache_deleted"] == 2
    assert body["discarded_ops"] == ["merge_images_oss", "post_process_stack_oss"]
    disc.assert_called_once()
    inval.assert_called_once()
    fake_state.drain_and_restart.assert_called_once()
    assert announced and announced[0][0] == "reload_data"


def test_discard_results_route_400_on_empty_selection():
    from web_server.backend import metadata
    with patch.object(metadata, "_resolve_entity", return_value=object()):
        r = _client().post(
            "/api/discard_results/mosaic/abc/P", json={"op_names": []},
        )
    assert r.status_code == 400


def test_discard_results_route_404_when_entity_missing():
    from web_server.backend import metadata
    with patch.object(metadata, "_resolve_entity", return_value=None):
        r = _client().post(
            "/api/discard_results/mosaic/x/P", json={"op_names": ["a"]},
        )
    assert r.status_code == 404


# --- invalidate_tap_snapshot (trace_paths) ---

def _stub_ctx(store):
    ctx = MagicMock()
    ctx.get.side_effect = lambda k, d=None: store.get(k, d)
    ctx.set.side_effect = lambda k, v: store.__setitem__(k, v)
    return ctx


def test_invalidate_tap_snapshot_removes_only_its_entry():
    from dependency_graph import trace_paths
    key = ("src", "stable", "content", "P")
    store = {"tap_snapshots": {key: "ref", ("other",): "x"}}
    with patch.object(trace_paths, "_snapshot_key", return_value=key), \
         patch.object(trace_paths, "context", _stub_ctx(store)):
        assert trace_paths.invalidate_tap_snapshot(("mosaic", "x", "P")) is True
    assert key not in store["tap_snapshots"]
    assert ("other",) in store["tap_snapshots"]   # siblings preserved


def test_invalidate_tap_snapshot_missing_key_is_noop():
    from dependency_graph import trace_paths
    store = {"tap_snapshots": {}}
    ctx = _stub_ctx(store)
    with patch.object(trace_paths, "_snapshot_key", return_value=("a",)), \
         patch.object(trace_paths, "context", ctx):
        assert trace_paths.invalidate_tap_snapshot(("mosaic", "x", "P")) is False
    ctx.set.assert_not_called()


def test_invalidate_tap_snapshot_unresolvable_key():
    from dependency_graph import trace_paths
    with patch.object(trace_paths, "_snapshot_key", return_value=None):
        assert trace_paths.invalidate_tap_snapshot(("mosaic", "x", "P")) is False
