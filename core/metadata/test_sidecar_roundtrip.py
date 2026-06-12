"""Unit tests for ImageMetadata sidecar serialization round-trip."""
import datetime
import numpy as np
import pytest
from pathlib import Path

from metadata.classes import ImageMetadata
from metadata.sidecar_io import (
    image_metadata_to_obj,
    image_metadata_from_obj,
    _serialize_header,
    _deserialize_header,
    read_defect_candidates,
    write_defect_candidates,
)


def _make_metadata(**overrides) -> ImageMetadata:
    defaults = dict(
        content_hash="abc123",
        stable_hash="stable456",
        filename=Path("/tmp/test.fits"),
        header={
            "IMAGETYP": "Bias",
            "DATE-OBS": datetime.datetime(2024, 1, 15, 20, 30, 0),
            "INSTRUME": "ZWO ASI6200MM Pro",
            "EXPOSURE": 300.0,
            "FILTER": "H",
            "CCD-TEMP": -20.0,
        },
        width_pixels=9576,
        height_pixels=6388,
        channels=1,
        dtype=np.dtype(np.uint16),
        plate_solution=None,
    )
    defaults.update(overrides)
    return ImageMetadata(**defaults)


def test_round_trip_preserves_fields():
    meta = _make_metadata()
    obj = image_metadata_to_obj(meta)
    restored = image_metadata_from_obj(obj, meta.filename)

    assert restored.content_hash == meta.content_hash
    assert restored.stable_hash == meta.stable_hash
    assert restored.width_pixels == meta.width_pixels
    assert restored.height_pixels == meta.height_pixels
    assert restored.channels == meta.channels
    assert restored.dtype == meta.dtype


def test_stable_hash_defaults_to_content_hash():
    meta = _make_metadata()
    obj = image_metadata_to_obj(meta)
    del obj["stable_hash"]
    restored = image_metadata_from_obj(obj, meta.filename)
    assert restored.stable_hash == meta.content_hash


def test_header_datetime_round_trip():
    dt = datetime.datetime(2024, 6, 15, 12, 30, 45)
    header = {"DATE-OBS": dt, "SIMPLE": True, "NAXIS": 2}
    serialized = _serialize_header(header)
    deserialized = _deserialize_header(serialized)
    assert deserialized["DATE-OBS"].year == 2024
    assert deserialized["DATE-OBS"].month == 6
    assert deserialized["SIMPLE"] is True
    assert deserialized["NAXIS"] == 2


def test_header_non_date_string_preserved():
    header = {"INSTRUME": "ZWO ASI6200MM Pro", "FILTER": "H"}
    serialized = _serialize_header(header)
    deserialized = _deserialize_header(serialized)
    assert deserialized["INSTRUME"] == "ZWO ASI6200MM Pro"
    assert deserialized["FILTER"] == "H"


def test_dtype_round_trip():
    for dt in [np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64)]:
        meta = _make_metadata(dtype=dt)
        obj = image_metadata_to_obj(meta)
        restored = image_metadata_from_obj(obj, meta.filename)
        assert restored.dtype == dt


def test_defect_candidates_not_in_image_metadata_obj():
    """`defect_candidates` no longer rides inside `image_metadata`; it
    lives in its own top-level sidecar section now."""
    meta = _make_metadata()
    obj = image_metadata_to_obj(meta)
    assert "defect_candidates" not in obj
    restored = image_metadata_from_obj(obj, meta.filename)
    assert not hasattr(restored, "defect_candidates")


def test_defect_candidates_section_round_trip(tmp_path):
    """`write_defect_candidates` / `read_defect_candidates` round-trip on
    the new top-level section."""
    fits = tmp_path / "frame.fits"
    fits.write_bytes(b"x")  # dummy data file, sidecar lives at .json sibling
    candidates = {1234: 8123.5, 99999: 12300.0, 4242: 5000.25}

    write_defect_candidates(fits, candidates)
    restored = read_defect_candidates(fits)
    assert restored == candidates


def test_defect_candidates_absent_returns_none(tmp_path):
    fits = tmp_path / "no_defects.fits"
    fits.write_bytes(b"x")
    assert read_defect_candidates(fits) is None


def test_read_defect_candidates_legacy_nested_fallback(tmp_path):
    """Pre-migration sidecars had `defect_candidates` nested inside the
    `image_metadata` section. The reader still finds it there until the
    migration script runs."""
    from file_io.sidecar import set_section

    fits = tmp_path / "legacy.fits"
    fits.write_bytes(b"x")
    set_section(fits, "image_metadata", {
        "content_hash": "abc",
        "stable_hash": "stable",
        "header": {},
        "width_pixels": 100,
        "height_pixels": 100,
        "channels": 1,
        "dtype": "uint16",
        "defect_candidates": {"42": 1.5, "100": 2.5},
    })

    restored = read_defect_candidates(fits)
    assert restored == {42: 1.5, 100: 2.5}


def test_write_defect_candidates_none_is_noop(tmp_path):
    fits = tmp_path / "none.fits"
    fits.write_bytes(b"x")
    write_defect_candidates(fits, None)
    assert read_defect_candidates(fits) is None


def test_master_integration_round_trip():
    meta = _make_metadata(master_integration_s=3600.5)
    obj = image_metadata_to_obj(meta)
    assert obj["master_integration_s"] == 3600.5
    restored = image_metadata_from_obj(obj, meta.filename)
    assert restored.master_integration_s == 3600.5


def test_master_integration_absent_remains_none():
    meta = _make_metadata()
    obj = image_metadata_to_obj(meta)
    assert "master_integration_s" not in obj
    restored = image_metadata_from_obj(obj, meta.filename)
    assert restored.master_integration_s is None


def test_unpickle_old_pickle_with_fewer_slots():
    """Old diskcache pickles predate `master_integration_s` (the last
    surviving optional field). Loading them via cloudpickle must NOT
    leave the new slot unset — otherwise the next pickle (e.g. ray
    transport) blows up. The __setstate__ override pads missing
    trailing slots with their declared defaults.
    """
    import pickle
    from dataclasses import fields
    from metadata.classes import ImageMetadata

    meta = _make_metadata()
    full_state = [getattr(meta, f.name) for f in fields(ImageMetadata)]

    # Simulate an old pickle whose state list is one field shorter —
    # i.e. predates master_integration_s.
    truncated = full_state[:-1]

    fresh = ImageMetadata.__new__(ImageMetadata)
    fresh.__setstate__(truncated)

    # The trailing new field must be filled with its default.
    assert fresh.master_integration_s is None
    # Old fields must round-trip exactly.
    assert fresh.content_hash == meta.content_hash
    assert fresh.width_pixels == meta.width_pixels

    # And critically: the restored instance must be re-pickleable.
    repickled = pickle.dumps(fresh)
    restored = pickle.loads(repickled)
    assert restored.master_integration_s is None
    assert restored.content_hash == meta.content_hash


def test_unpickle_newer_pickle_with_extra_trailing_slot():
    """Symmetric to the previous test: a pickle written by an OLDER
    code version that still had `defect_candidates` as a trailing slot
    must load cleanly under the new code where that slot has been
    removed. The __setstate__ override truncates extra trailing
    entries to match the current field list.
    """
    import pickle
    from dataclasses import fields
    from metadata.classes import ImageMetadata

    meta = _make_metadata()
    field_names = [f.name for f in fields(ImageMetadata)]
    current_state = [getattr(meta, f.name) for f in fields(ImageMetadata)]

    # Simulate an older pickle that still carried `defect_candidates`
    # at the position right before `master_integration_s`. We mimic
    # this by inserting one extra entry at that exact position into
    # the state list — looking the position up by name rather than
    # by negative index so new fields appended later (e.g.
    # `statistics`) don't move the insertion target. The
    # __setstate__ should truncate to the current field count.
    extra = {42: 1.0, 100: 2.0}  # what defect_candidates would have been
    master_int_idx = field_names.index("master_integration_s")
    legacy_state = (
        current_state[:master_int_idx]
        + [extra]
        + current_state[master_int_idx:]
    )

    fresh = ImageMetadata.__new__(ImageMetadata)
    fresh.__setstate__(legacy_state)

    # Current fields round-trip correctly.
    assert fresh.content_hash == meta.content_hash
    assert fresh.master_integration_s == meta.master_integration_s
    # Re-pickle works.
    pickle.loads(pickle.dumps(fresh))
