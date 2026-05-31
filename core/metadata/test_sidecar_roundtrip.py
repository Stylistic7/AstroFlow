"""Unit tests for ImageMetadata sidecar serialization round-trip."""
import datetime
import numpy as np
from pathlib import Path

from metadata.classes import ImageMetadata
from metadata.sidecar_io import image_metadata_to_obj, image_metadata_from_obj, _serialize_header, _deserialize_header


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


def test_defect_candidates_round_trip():
    candidates = {1234: 8123.5, 99999: 12300.0, 4242: 5000.25}
    meta = _make_metadata(defect_candidates=candidates)
    obj = image_metadata_to_obj(meta)
    # JSON requires string keys; verify intermediate form and final restoration.
    assert obj["defect_candidates"] == {"1234": 8123.5, "99999": 12300.0, "4242": 5000.25}
    restored = image_metadata_from_obj(obj, meta.filename)
    assert restored.defect_candidates == candidates


def test_defect_candidates_absent_remains_none():
    meta = _make_metadata()
    obj = image_metadata_to_obj(meta)
    assert "defect_candidates" not in obj
    restored = image_metadata_from_obj(obj, meta.filename)
    assert restored.defect_candidates is None


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
    """Old diskcache pickles predate defect_candidates + master_integration_s.
    Loading them via cloudpickle must NOT leave the new slots unset —
    otherwise the next pickle (e.g. ray.get_matching transport) blows up
    with `AttributeError: 'ImageMetadata' object has no attribute
    'defect_candidates'`. The __setstate__ override pads missing trailing
    slots with their declared defaults.
    """
    import pickle
    from dataclasses import fields
    from metadata.classes import ImageMetadata

    meta = _make_metadata()
    full_state = [getattr(meta, f.name) for f in fields(ImageMetadata)]

    # Simulate an old pickle whose state list is two fields shorter —
    # i.e. predates defect_candidates and master_integration_s.
    truncated = full_state[:-2]

    # Build a fresh instance and apply the truncated state directly,
    # mimicking what cloudpickle does on restore.
    fresh = ImageMetadata.__new__(ImageMetadata)
    fresh.__setstate__(truncated)

    # New fields must be set to their defaults, not missing.
    assert fresh.defect_candidates is None
    assert fresh.master_integration_s is None
    # Old fields must round-trip exactly.
    assert fresh.content_hash == meta.content_hash
    assert fresh.width_pixels == meta.width_pixels

    # And critically: the restored instance must be re-pickleable.
    repickled = pickle.dumps(fresh)
    restored = pickle.loads(repickled)
    assert restored.defect_candidates is None
    assert restored.master_integration_s is None
    assert restored.content_hash == meta.content_hash
