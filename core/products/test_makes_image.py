"""The image-producer lifecycle (`products.registry.makes_image_wrap`).

A producer writes the scratch path it is handed (`out`) and returns
`Produced(out, header=, sections=)`; the framework stamps, builds the pyramid,
records the sidecar and lands the file with one `os.replace`. Pinned here:

  * the final path is touched exactly once, last — a reader sees the previous
    product or the new one, never a partial (the 2026-08-14 γ Cyg master served
    36,150 broken tiles while its pyramid was still building);
  * a failed pyramid still publishes the planar payload (the master is the
    expensive part);
  * a body that raises or returns None publishes nothing and leaves nothing;
  * the sidecar carries the returned sections before the payload lands;
  * a body may not write a sidecar of its own.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from file_io.sidecar import get_section, set_section, sidecar_path_for
from file_io.types import Produced, TiffFilePath
from products import registry


def _write_float_tiff(path, a):
    import pyvips
    h, w = a.shape[:2]
    pyvips.Image.new_from_memory(np.ascontiguousarray(a).tobytes(), w, h, 1, "float").tiffsave(str(path))


A = (np.arange(300 * 200, dtype="float32").reshape(200, 300) / 1e4)


def _producer(store, body, makes_tiff=True):
    """`body(out)` wrapped as a display producer whose product is
    `store/abc123.tiff`."""
    def op(x, *, out: TiffFilePath) -> Produced:
        return body(out)
    return registry.makes_image_wrap(
        op, filename_fn=lambda *a, **k: TiffFilePath(store / "abc123.tiff"),
        canonical_call_hash=lambda *a, **k: "abc123",
        makes_tiff=makes_tiff, image_type=None)


@pytest.fixture
def store(tmp_path):
    return tmp_path / "store"


def test_body_writes_a_scratch_beside_the_product_and_the_product_lands_once(store, monkeypatch):
    seen = {}

    def body(out):
        seen["out"] = out
        seen["final_during_body"] = (store / "abc123.tiff").exists()
        _write_float_tiff(out, A)
        return Produced(out)

    replaced = []
    real = os.replace
    monkeypatch.setattr(registry.os, "replace", lambda a, b: (replaced.append(Path(b).name), real(a, b)))
    final = _producer(store, body)("x")

    out = seen["out"]
    assert out.parent == store and out.name.startswith(".abc123.partial.") and out.suffix == ".tiff"
    assert isinstance(out, TiffFilePath)
    assert not seen["final_during_body"]
    assert replaced.count("abc123.tiff") == 1               # one landing, nothing parked
    assert replaced[-1] == "abc123.tiff"                    # and it is the last thing done
    assert final == store / "abc123.tiff" and final.exists()
    assert sorted(f.name for f in store.iterdir()) == ["abc123.tiff", "abc123.tiff.json"]


def test_a_tiff_display_product_is_its_pyramid_and_pixel_identical(store):
    import pyvips
    from file_io.planar import is_tiled_tiff
    final = _producer(store, lambda out: (_write_float_tiff(out, A), Produced(out))[1])("x")
    assert is_tiled_tiff(final)
    page0 = pyvips.Image.tiffload(str(final), page=0)
    got = np.ndarray(buffer=page0.write_to_memory(), dtype="float32", shape=(page0.height, page0.width))
    assert np.array_equal(got, A)
    assert not final.with_name("abc123_tiles.tiff").exists()


def test_a_non_display_product_stays_planar(store):
    from file_io.planar import is_tiled_tiff
    final = _producer(store, lambda out: (_write_float_tiff(out, A), Produced(out))[1], makes_tiff=False)("x")
    assert final.exists() and not is_tiled_tiff(final)


def test_a_failed_pyramid_still_publishes_the_planar_master(store, monkeypatch):
    from previews import create_tile_pyramid as ctp

    def boom(*a, **k):
        raise MemoryError("OOM")
    monkeypatch.setattr(ctp, "default_make_tile_pyramid", boom)
    final = _producer(store, lambda out: (_write_float_tiff(out, A), Produced(out))[1])("x")
    assert final.exists()
    assert sorted(f.name for f in store.iterdir()) == ["abc123.tiff", "abc123.tiff.json"]


def test_a_body_that_raises_publishes_nothing_and_keeps_the_previous_product(store):
    store.mkdir(parents=True)
    (store / "abc123.tiff").write_bytes(b"previous")

    def body(out):
        _write_float_tiff(out, A)
        raise RuntimeError("tool crashed")
    with pytest.raises(RuntimeError):
        _producer(store, body)("x")
    assert (store / "abc123.tiff").read_bytes() == b"previous"
    assert [f.name for f in store.iterdir()] == ["abc123.tiff"]


def test_a_body_returning_none_publishes_nothing(store):
    assert _producer(store, lambda out: None)("x") is None
    assert list(store.iterdir()) == []


def test_returned_sections_and_header_are_recorded(store):
    from file_io.generic import read_header

    def body(out):
        _write_float_tiff(out, A)
        return Produced(out, header={"PHOTREF": 2.5}, sections={"report": {"n": 3}})
    final = _producer(store, body)("x")
    assert get_section(final, "report") == {"n": 3}
    hdr = read_header(final)
    assert float(hdr["PHOTREF"]) == 2.5 and str(hdr["CALLHASH"]) == "abc123"


def test_sections_merge_into_the_existing_sidecar_and_old_metadata_is_dropped(store):
    """Merged, not replaced: a plate solution an annotating node recorded for
    this path survives a rebuild. The measured `image_metadata` describes the
    OLD bytes, so it goes."""
    store.mkdir(parents=True)
    old = store / "abc123.tiff"
    old.write_bytes(b"old")
    set_section(old, "plate_solution", {"data": "kept"})
    set_section(old, "image_metadata", {"stale": True})
    final = _producer(store, lambda out: (_write_float_tiff(out, A), Produced(out, sections={"r": 1}))[1])("x")
    assert get_section(final, "plate_solution") == {"data": "kept"}
    assert get_section(final, "image_metadata") is None
    assert get_section(final, "r") == 1


def test_a_body_may_not_write_its_own_sidecar(store):
    def body(out):
        _write_float_tiff(out, A)
        set_section(out, "report", 1)
        return Produced(out)
    with pytest.raises(AssertionError, match="Produced\\(sections"):
        _producer(store, body)("x")
    assert not (store / "abc123.tiff").exists()
    assert not any(sidecar_path_for(f).exists() for f in store.iterdir() if f.name.startswith("."))


def test_a_body_must_return_its_own_out(store, tmp_path):
    def body(out):
        other = tmp_path / "elsewhere.tiff"
        _write_float_tiff(other, A)
        return Produced(other)
    with pytest.raises(AssertionError, match="Produced\\(out"):
        _producer(store, body)("x")


def test_the_wrapper_hides_out_from_the_signature():
    import inspect
    w = _producer(Path("/nonexistent"), lambda out: None)
    assert list(inspect.signature(w).parameters) == ["x"]
