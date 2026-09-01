"""The snapshot-lag guard in `resolve_product` (2026-08-31).

A build-time tap snapshot can lag a replan; then the predicted "current"
master is an OLD generation whose file exists, and the resolver used to
report it "live" (the boot-16 'Linear SHO current while the current file
is absent' confusion). The guard: a NEWER completed entry in the version
log outranks a stale-snapshot live — served with the out_of_date badge."""
import types

import catalog.queries as q
from catalog import Product


PROD = Product("mosaic", "m1", "Pipe", "Linear SHO", "RGB")


def _patch(monkeypatch, *, predicted, viewable, log_entries):
    monkeypatch.setattr(q, "predict_path", lambda p, a=(): predicted)
    monkeypatch.setattr(q, "viewable_from_master",
                        lambda m: viewable.get(str(m)) if m else None)
    import products.version_log as vl
    monkeypatch.setattr(vl, "versions", lambda k: log_entries)
    monkeypatch.setattr(vl, "call_hash_of", lambda p: str(p).split("/")[-1].split(".")[0])
    monkeypatch.setattr(vl, "key_for", lambda p: ("k",))


def test_fresh_snapshot_live(monkeypatch):
    _patch(monkeypatch, predicted="/x/new123.tiff",
           viewable={"/x/new123.tiff": "/x/new123.tiff"},
           log_entries=[{"call_hash": "new123", "path": "/x/new123.tiff"}])
    path, status = q.resolve_product(PROD)
    assert status == "live"


def test_stale_snapshot_outranked_by_newer_log_entry(monkeypatch):
    # snapshot predicts OLD (exists); log's newest is a DIFFERENT completed
    # generation -> serve the newest, badged out_of_date.
    _patch(monkeypatch, predicted="/x/old111.tiff",
           viewable={"/x/old111.tiff": "/x/old111.tiff",
                     "/x/new222.tiff": "/x/new222.tiff"},
           log_entries=[{"call_hash": "old111", "path": "/x/old111.tiff"},
                        {"call_hash": "new222", "path": "/x/new222.tiff"}])
    path, status = q.resolve_product(PROD)
    assert status == "out_of_date"
    assert "new222" in str(path)


def test_newer_log_entry_not_on_disk_keeps_live(monkeypatch):
    # newest log entry's file was swept -> the predicted live stands.
    _patch(monkeypatch, predicted="/x/old111.tiff",
           viewable={"/x/old111.tiff": "/x/old111.tiff"},
           log_entries=[{"call_hash": "old111", "path": "/x/old111.tiff"},
                        {"call_hash": "new222", "path": "/x/new222.tiff"}])
    path, status = q.resolve_product(PROD)
    assert status == "live"
    assert "old111" in str(path)
