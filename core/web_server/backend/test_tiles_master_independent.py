"""The existence-gated PRIMARY path (`catalog.master_path`) that the
download / open endpoints rely on, as distinct from the servable
pyramid-or-master resolver (`catalog.resolve_product`, pinned in
`catalog/test_queries.py`): the master can be reclaimed mid-pipeline while
its pyramid stays, and each caller must get the answer it asked for.

Paths come from the catalog's per-entity tap filing (`catalog.predict_path`),
not a stateful index; these tests patch that single predictor.
"""
from pathlib import Path


def _patch_predict(monkeypatch, returned):
    """Make `catalog.predict_path` (the single path predictor) answer `returned`
    (a path string, or None for "the trace has no such tap")."""
    import catalog.queries as cat
    monkeypatch.setattr(
        cat, "predict_path",
        lambda product: Path(returned) if returned is not None else None)


class TestMasterPathExistenceGate:
    """`catalog.master_path` (the existence-gated primary) MUST keep its
    404-on-missing semantics — file download / open endpoints rely on it."""

    @staticmethod
    def _product():
        from catalog import Product
        return Product("mosaic", "id", "PI", "Merged", "H")

    def test_returns_none_when_master_missing(self, tmp_path, monkeypatch):
        master = tmp_path / "ephemeral_reclaimed.fits"  # never written
        _patch_predict(monkeypatch, str(master))
        from catalog import master_path
        assert master_path(self._product()) is None

    def test_returns_path_when_master_exists(self, tmp_path, monkeypatch):
        master = tmp_path / "alive.fits"
        master.write_bytes(b"x")
        _patch_predict(monkeypatch, str(master))
        from catalog import master_path
        assert master_path(self._product()) == master

    def test_returns_none_when_trace_has_no_tap(self, monkeypatch):
        """When the trace doesn't predict a path for `(display, filter)` —
        e.g. the pipeline never produces that tap — there is nothing to gate."""
        _patch_predict(monkeypatch, None)
        from catalog import master_path
        assert master_path(self._product()) is None
