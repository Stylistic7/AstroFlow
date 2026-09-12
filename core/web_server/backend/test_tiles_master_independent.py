"""Tests that the tile endpoint serves `.tiff` siblings independently of
the master FITS/XISF on disk.

After `discard_sidecar` was removed and `ephemeral_intermediates`
introduced, the master files for placed/merged outputs can be reclaimed
mid-pipeline while their tile pyramids stay. The tile endpoint must
still resolve in that case — that's the whole reason the catalog has a
viewable (`viewable_path`, pyramid-or-master) resolver distinct from the
primary-only one (`master_path`).

Paths come from the catalog's per-entity tap filing (`catalog.predict_path`),
not a stateful index; these tests patch that single predictor and exercise
the existence-gating on top of it.
"""
from pathlib import Path

import pytest


def _patch_predict(monkeypatch, returned):
    """Make `catalog.predict_path` (the single path predictor) answer `returned`
    (a path string, or None for "the trace has no such tap")."""
    import catalog.queries as cat
    monkeypatch.setattr(
        cat, "predict_path",
        lambda product: Path(returned) if returned is not None else None)


class TestTilePyramidPathForTap:
    """`_tile_pyramid_path_for_tap` must return the `<stem>_tiles.tiff`
    display-pyramid path when it exists, regardless of whether the master
    FITS/XISF/TIFF deliverable does."""

    def _patch_tap_path(self, monkeypatch, returned_path: str):
        _patch_predict(monkeypatch, returned_path)

    def test_resolves_when_tiff_exists_and_master_does_not(
        self, tmp_path, monkeypatch,
    ):
        from previews.create_tile_pyramid import pyramid_path_for
        master = tmp_path / "merged_abc.fits"  # NOT created
        tiff = pyramid_path_for(master)
        tiff.write_bytes(b"fake pyramid")

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "PI Downscaled", "Merged 5", "mosaic", "abc", "H",
        )
        assert result == str(tiff)

    def test_resolves_when_both_master_and_tiff_exist(
        self, tmp_path, monkeypatch,
    ):
        """Common case: master not yet reclaimed; tile pyramid sits beside it."""
        from previews.create_tile_pyramid import pyramid_path_for
        master = tmp_path / "merged_xyz.fits"
        master.write_bytes(b"fake fits")
        tiff = pyramid_path_for(master)
        tiff.write_bytes(b"fake pyramid")

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "PI Downscaled", "Merged", "mosaic", "xyz", "H",
        )
        assert result == str(tiff)

    def test_returns_none_when_tiff_missing(
        self, tmp_path, monkeypatch,
    ):
        master = tmp_path / "lonely.fits"
        master.write_bytes(b"x")
        # No tiff sibling.

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "PI", "Merged", "mosaic", "id", "H",
        )
        assert result is None

    def test_resolves_tiles_sibling_when_master_is_itself_tiff(
        self, tmp_path, monkeypatch,
    ):
        """A `.tiff` deliverable (place_image, color images, Subframe
        mosaics) has its display pyramid at the separate `<stem>_tiles.tiff`
        sibling — the pyramid never aliases the user's deliverable, so the
        resolver returns the `_tiles.tiff`, NOT the master itself."""
        from previews.create_tile_pyramid import pyramid_path_for
        master = tmp_path / "subframe.tiff"
        master.write_bytes(b"fake tiff deliverable")
        tiff = pyramid_path_for(master)  # subframe_tiles.tiff
        tiff.write_bytes(b"fake pyramid")

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "Subframe", "Mosaic", "mosaic", "id", "H",
        )
        assert result == str(tiff)

    @staticmethod
    def _write_tiff(path, fmt):
        """A REAL tiff — the resolver reads its geometry to decide whether it
        needs a pyramid, so a placeholder byte-string would read as unreadable
        (and correctly refuse to serve)."""
        import numpy as np
        import pyvips
        rng = np.random.default_rng(0)
        if fmt == "uchar":
            a = (rng.random((16, 16, 3)) * 255).astype("uint8")
        else:
            a = (rng.random((16, 16, 3)) * 0.1).astype("float32")
        pyvips.Image.new_from_memory(a.tobytes(), 16, 16, 3, fmt).tiffsave(str(path))
        return path

    def test_falls_back_to_master_tiff_when_no_tiles_sibling(
        self, tmp_path, monkeypatch,
    ):
        """Already-display-ready uint8 tiffs (Subframe previews) have no
        separate `_tiles.tiff` — they serve themselves, so the resolver
        falls back to the master tiff when no pyramid sibling exists."""
        master = self._write_tiff(tmp_path / "subframe_uint8.tiff", "uchar")
        # No `_tiles.tiff` sibling built for it.

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "Subframe", "Mosaic", "mosaic", "id", "H",
        )
        assert result == str(master)

    def test_float_master_with_no_tiles_sibling_is_refused(
        self, tmp_path, monkeypatch,
    ):
        """The 2026-08-14 regression. A float master is one page and
        strip-based; tiling it yields full resolution at every zoom level, so
        the resolver must report "not built yet" instead of falling back."""
        master = self._write_tiff(tmp_path / "linear_rgb.tiff", "float")

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "Linear RGB", "Mosaic", "mosaic", "id", "RGB",
        )
        assert result is None


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
