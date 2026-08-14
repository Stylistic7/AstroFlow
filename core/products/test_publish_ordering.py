"""The pyramid lands BEFORE the payload — 2026-08-14.

A γ Cyg `Linear RGB` master was published at its final path while its tile
pyramid took another ~15 minutes to build. For that whole window the product
was RESOLVABLE BUT UNTILEABLE, and on a rebuild the new payload had already
destroyed the working previous one. The viewer served 36,150 broken tiles.

`_publish_with_presentations` parks the payload under a dot-name, builds the
pyramid from there, lands the PYRAMID first and the PAYLOAD last. Two
guarantees are load-bearing and pinned here:

  * payload present => pyramid present (ordering);
  * the payload is republished even when the pyramid build raises — otherwise
    a failed pyramid would strand an expensive master under its staging name
    and the product would vanish, strictly worse than a missing pyramid.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from products import registry


@pytest.fixture
def master(tmp_path):
    """A float TIFF standing in for a deliverable (float => needs a pyramid)."""
    import pyvips
    a = (np.random.default_rng(0).random((64, 64, 3)) * 0.1).astype("float32")
    p = tmp_path / "store" / "abc123.tiff"
    p.parent.mkdir(parents=True)
    pyvips.Image.new_from_memory(a.tobytes(), 64, 64, 3, "float").tiffsave(str(p))
    return p


def _run(master, monkeypatch, pyramid_side_effect=None, record=None):
    """Drive _publish_with_presentations over `master`, optionally making the
    pyramid build explode, recording os.replace order."""
    from previews import create_tile_pyramid as ctp

    def fake_build(src, dest):
        if record is not None:
            record.append(("build_from", Path(src).name))
        if pyramid_side_effect is not None:
            raise pyramid_side_effect
        Path(dest).write_bytes(b"pyramid")

    monkeypatch.setattr(ctp, "default_make_tile_pyramid", fake_build)
    if record is not None:
        real = os.replace

        def spy(a, b):
            record.append(("replace_to", Path(b).name))
            return real(a, b)

        monkeypatch.setattr(registry.os, "replace", spy)
    monkeypatch.setattr(registry, "_stamp_header", lambda *a, **k: None)
    registry._publish_with_presentations(master, "abc123", None, ())


class TestOrdering:
    def test_pyramid_lands_before_payload(self, master, monkeypatch):
        record = []
        _run(master, monkeypatch, record=record)
        replaces = [n for kind, n in record if kind == "replace_to"]
        # unpublish (dot-name), then the payload republish last
        assert replaces[0].startswith(".abc123.staging"), replaces
        assert replaces[-1] == "abc123.tiff", replaces
        pyr = master.with_name("abc123_tiles.tiff")
        assert pyr.exists(), "pyramid was not written"

    def test_pyramid_is_built_from_the_parked_payload(self, master, monkeypatch):
        record = []
        _run(master, monkeypatch, record=record)
        built = [n for kind, n in record if kind == "build_from"]
        assert built and built[0].startswith(".abc123.staging"), built

    def test_staging_name_preserves_the_suffix(self, master, monkeypatch):
        """Header reads, update_header and needs_display_pyramid all dispatch
        on extension — a parked `.tiff` must stay a `.tiff`."""
        record = []
        _run(master, monkeypatch, record=record)
        parked = [n for kind, n in record if kind == "replace_to"][0]
        assert parked.endswith(".tiff"), parked

    def test_payload_is_republished_even_if_the_pyramid_raises(
            self, master, monkeypatch):
        _run(master, monkeypatch, pyramid_side_effect=MemoryError("OOM"))
        assert master.exists(), \
            "payload stranded under its staging name — product would vanish"
        assert not any(p.name.startswith(".abc123.staging")
                       for p in master.parent.iterdir()), "staging leaked"

    def test_no_pyramid_needed_still_republishes(self, tmp_path, monkeypatch):
        """A uint8 tiff is its own display artifact; it must survive the round
        trip untouched rather than being parked and forgotten."""
        import pyvips
        a = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype("uint8")
        p = tmp_path / "u.tiff"
        pyvips.Image.new_from_memory(a.tobytes(), 32, 32, 3, "uchar").tiffsave(str(p))
        before = p.read_bytes()
        monkeypatch.setattr(registry, "_stamp_header", lambda *a, **k: None)
        registry._publish_with_presentations(p, "u", None, ())
        assert p.exists() and p.read_bytes() == before


class TestNonFileResults:
    def test_ndarray_result_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(registry, "_stamp_header", lambda *a, **k: None)
        monkeypatch.setattr(registry, "_generate_presentations",
                            lambda *a, **k: None)
        registry._publish_with_presentations(np.zeros((4, 4)), "h", None, ())

    def test_result_path_unwraps_the_usual_shapes(self, tmp_path):
        p = tmp_path / "x.tiff"
        p.write_bytes(b"")
        assert registry._result_path(p) == p
        assert registry._result_path(np.zeros(3)) is None
