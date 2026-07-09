"""Tests that the tile endpoint serves `.tiff` siblings independently of
the master FITS/XISF on disk.

After `discard_sidecar` was removed and `ephemeral_intermediates`
introduced, the master files for placed/merged outputs can be reclaimed
mid-pipeline while their tile pyramids stay. The tile endpoint must
still resolve in that case — that's the whole reason `tap_path_for`
was split into existence-gated + unchecked variants.

Paths now come from the pipeline trace (`dependency_graph.trace_paths.
tap_path_for`), not a stateful index; these tests patch that single
predictor and exercise the existence-gating on top of it.
"""
from pathlib import Path

import pytest


class TestTilePyramidPathForTap:
    """`_tile_pyramid_path_for_tap` must return the `<stem>_tiles.tiff`
    display-pyramid path when it exists, regardless of whether the master
    FITS/XISF/TIFF deliverable does."""

    def _patch_tap_path(self, monkeypatch, returned_path: str):
        """Make the trace-based `tap_path_for` (the single path predictor)
        return our test path. Both `metadata.tap_path_for` and
        `_tap_path_unchecked` route through it via a lazy import."""
        import dependency_graph.trace_paths as tp
        monkeypatch.setattr(
            tp, "tap_path_for",
            lambda item, display, filter: Path(returned_path),
        )

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

    def test_falls_back_to_master_tiff_when_no_tiles_sibling(
        self, tmp_path, monkeypatch,
    ):
        """Already-display-ready uint8 tiffs (Subframe previews) have no
        separate `_tiles.tiff` — they serve themselves, so the resolver
        falls back to the master tiff when no pyramid sibling exists."""
        master = tmp_path / "subframe_uint8.tiff"
        master.write_bytes(b"fake uint8 subframe tiff")
        # No `_tiles.tiff` sibling built for it.

        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.tiles import _tile_pyramid_path_for_tap
        result = _tile_pyramid_path_for_tap(
            "Subframe", "Mosaic", "mosaic", "id", "H",
        )
        assert result == str(master)


class TestTapPathForExistenceGate:
    """`tap_path_for` (the existence-gated public API) MUST keep its
    404-on-missing semantics — file download endpoints rely on it."""

    def _patch_tap_path(self, monkeypatch, returned_path: str):
        import dependency_graph.trace_paths as tp
        monkeypatch.setattr(
            tp, "tap_path_for",
            lambda item, display, filter: Path(returned_path),
        )

    def test_returns_none_when_master_missing(
        self, tmp_path, monkeypatch,
    ):
        master = tmp_path / "ephemeral_reclaimed.fits"  # never written
        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.metadata import tap_path_for
        assert tap_path_for(
            "PI Downscaled", "Merged 5", "mosaic", "id", "H",
        ) is None

    def test_returns_path_when_master_exists(
        self, tmp_path, monkeypatch,
    ):
        master = tmp_path / "alive.fits"
        master.write_bytes(b"x")
        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.metadata import tap_path_for
        assert tap_path_for("PI", "Merged", "mosaic", "id", "H") == str(master)

    def test_unchecked_variant_returns_path_even_when_missing(
        self, tmp_path, monkeypatch,
    ):
        master = tmp_path / "vapor.fits"  # never written
        self._patch_tap_path(monkeypatch, str(master))
        from web_server.backend.metadata import _tap_path_unchecked
        assert _tap_path_unchecked(
            "PI", "Merged", "mosaic", "id", "H",
        ) == str(master)

    def test_unchecked_variant_returns_none_when_trace_has_no_tap(
        self, monkeypatch,
    ):
        """When the trace doesn't predict a path for `(display, filter)` —
        e.g. the pipeline never produces that tap — the predictor returns
        None and `_tap_path_unchecked` passes it through as None (no
        sentinel string to filter; there's no index anymore)."""
        import dependency_graph.trace_paths as tp
        monkeypatch.setattr(
            tp, "tap_path_for",
            lambda item, display, filter: None,
        )
        from web_server.backend.metadata import _tap_path_unchecked, tap_path_for
        assert _tap_path_unchecked("PI", "Merged", "mosaic", "id", "H") is None
        assert tap_path_for("PI", "Merged", "mosaic", "id", "H") is None
