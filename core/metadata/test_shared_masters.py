"""Reading a shared master folder (`metadata.shared_masters`): the scan
lists every image and leaves a sync tool's temporaries out; a file without a
current sidecar is measured in place; a master is admissible once whole,
solved, a single-band light — whoever made it."""
from pathlib import Path

from metadata.shared_masters import (
    SETTLED_AFTER_S, admissible_masters, compute_shared_master, measurements_due,
    scan_shared_dir, sidecar_is_current,
)
from testing.shared_masters import solved_master_metadata, write_shared_master


def test_scan_lists_every_image_and_skips_dot_files(tmp_path):
    write_shared_master(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="1" * 16)
    (tmp_path / "loose.fits").write_bytes(b"no sidecar yet")
    (tmp_path / ".syncthing.b.fits.tmp").write_bytes(b"arriving")
    (tmp_path / "notes.json").write_text("{}")
    (tmp_path / "a (conflicted copy).fits.json").write_text("{}")
    record = scan_shared_dir(tmp_path)
    assert [rel for rel, _, _ in record.listing] == ["a.fits", "loose.fits"]
    assert record.fingerprint == scan_shared_dir(tmp_path).fingerprint


def test_the_fingerprint_moves_when_a_sidecar_changes(tmp_path):
    write_shared_master(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="1" * 16)
    before = scan_shared_dir(tmp_path).fingerprint
    from file_io.sidecar import set_section
    import os, time
    set_section(tmp_path / "a.fits", "shared", {"observer": "W"})
    os.utime(tmp_path / "a.fits.json", (time.time() + 5, time.time() + 5))
    assert scan_shared_dir(tmp_path).fingerprint != before


def test_a_sidecar_is_current_when_it_recorded_the_files_size(tmp_path):
    write_shared_master(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="1" * 16)
    size = (tmp_path / "a.fits").stat().st_size
    assert sidecar_is_current(tmp_path / "a.fits", size)
    assert not sidecar_is_current(tmp_path / "a.fits", size + 100)
    (tmp_path / "b.fits").write_bytes(b"x" * 10)
    assert not sidecar_is_current(tmp_path / "b.fits", 10), "no sidecar at all"


def _record(root, **overrides):
    r = scan_shared_dir(root)
    return r if not overrides else type(r)(**{**r.__dict__, **overrides})


def test_what_a_scan_must_measure(tmp_path):
    import os, time
    write_shared_master(tmp_path / "described.fits", call_hash="a" * 16, stable_hash="1" * 16)
    (tmp_path / "fresh.fits").write_bytes(b"x" * 10)        # no sidecar, just arrived
    (tmp_path / "old.fits").write_bytes(b"y" * 10)          # no sidecar, long settled
    os.utime(tmp_path / "old.fits", (time.time() - 2 * SETTLED_AFTER_S, time.time() - 2 * SETTLED_AFTER_S))
    now = time.time()

    # First sight: only the settled file is measured; the fresh one waits.
    record = scan_shared_dir(tmp_path)
    due = measurements_due(tmp_path, None, record, now)
    assert [(Path(p).name, v) for p, _, _, v in due] == [("old.fits", False)]

    # Next scan, nothing moved: the fresh file has now sat unchanged — measure it.
    due = measurements_due(tmp_path, record, scan_shared_dir(tmp_path), now)
    assert sorted((Path(p).name, v) for p, _, _, v in due) == [("fresh.fits", False), ("old.fits", False)]

    # A described file moved under its sidecar: verify by hash, do not trust.
    with open(tmp_path / "described.fits", "ab") as f:
        f.write(b"")
    os.utime(tmp_path / "described.fits", ns=(1, record.listing[0][2] + 10 ** 9))
    due = measurements_due(tmp_path, record, scan_shared_dir(tmp_path), now)
    assert ("described.fits", True) in [(Path(p).name, v) for p, _, _, v in due]


def test_a_changed_file_is_measured_again_only_if_its_pixels_changed(tmp_path, monkeypatch):
    import metadata.image_metadata as im
    import metadata.shared_masters as sm
    data = tmp_path / "a.fits"
    meta = write_shared_master(data, call_hash="a" * 16, stable_hash="1" * 16)
    del meta.header["CALLHASH"]           # another program's master: content hash = pixels
    import dataclasses
    from file_io.generic import hash_of_image_data
    write_shared_master(data, dataclasses.replace(meta, content_hash=hash_of_image_data(data)))
    seen = []
    monkeypatch.setattr(im.compute_image_metadata, "_function",
                        lambda path, plate_solve=None: seen.append((path, (tmp_path / "a.fits.json").exists())))
    size = data.stat().st_size
    compute_shared_master._function(str(data), size, 0, True)
    assert seen == [], "same pixels: the sidecar stands, nothing is written"
    # the pixels changed (same size): the stale sidecar goes and the file is measured
    from astropy.io import fits
    with fits.open(data, mode="update") as h:
        h[0].data[0, 0] = 1.0
    compute_shared_master._function(str(data), data.stat().st_size, 0, True)
    assert seen == [(data, False)]
    # a sidecar recording other bytes is dropped without a hash
    seen.clear()
    with open(data, "ab") as f:
        f.write(b"\0" * 100)
    compute_shared_master._function(str(data), data.stat().st_size, 0, False)
    assert seen == [(data, False)]


def test_whole_solved_single_band_light_masters_are_admissible_whoever_made_them(tmp_path):
    ours = write_shared_master(tmp_path / "ours.fits", call_hash="a" * 16, stable_hash="1" * 16)
    theirs = solved_master_metadata(tmp_path / "theirs.fits", call_hash="b" * 16, stable_hash="2" * 16)
    for k in ("PIPEHASH", "CALLHASH", "STBLHASH"):
        del theirs.header[k]
    write_shared_master(tmp_path / "theirs.fits", theirs)
    flat = solved_master_metadata(tmp_path / "flat.fits", call_hash="c" * 16, stable_hash="3" * 16)
    flat.header["IMAGETYP"] = "master flat"
    write_shared_master(tmp_path / "flat.fits", flat)
    rgb = solved_master_metadata(tmp_path / "rgb.fits", call_hash="e" * 16, stable_hash="5" * 16)
    import dataclasses
    write_shared_master(tmp_path / "rgb.fits", dataclasses.replace(rgb, channels=3))
    write_shared_master(tmp_path / "partial.fits", call_hash="d" * 16, stable_hash="4" * 16)
    with open(tmp_path / "partial.fits", "ab") as f:
        f.write(b"\0" * 100)   # the sync tool is still delivering it
    (tmp_path / "unmeasured.fits").write_bytes(b"no sidecar")

    listing = scan_shared_dir(tmp_path).listing
    masters = admissible_masters(tmp_path, listing)
    assert set(masters) == {ours.content_hash, theirs.content_hash}
    assert masters[theirs.content_hash].plate_solution is not None
    assert masters[ours.content_hash].filename == tmp_path / "ours.fits"


def test_a_shared_master_without_imagetyp_is_a_light_and_a_calibration_master_never_is(tmp_path):
    from metadata.shared_masters import declares_light
    assert declares_light({"FILTER": "H"})                     # survey products carry none
    assert declares_light({"IMAGETYP": "Light Frame"})
    assert not declares_light({"IMAGETYP": "master flat"})
    survey = solved_master_metadata(tmp_path / "survey.fits", call_hash="f" * 16, stable_hash="6" * 16)
    del survey.header["IMAGETYP"]
    write_shared_master(tmp_path / "survey.fits", survey)
    assert set(admissible_masters(tmp_path, scan_shared_dir(tmp_path).listing)) == {survey.content_hash}
