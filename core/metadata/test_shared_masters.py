"""Reading a shared master folder (`metadata.shared_masters`): the scan
lists every image and leaves a sync tool's temporaries out; a file without a
current sidecar is measured in place; a master is admissible once whole,
solved, a single-band light — whoever made it."""
from pathlib import Path

from metadata.shared_masters import (
    admissible_masters, compute_shared_master, scan_shared_dir, sidecar_is_current,
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


def test_a_sidecar_is_current_only_for_the_bytes_it_recorded(tmp_path):
    import os
    from file_io.sidecar import set_section
    from metadata.shared_masters import MEASURED
    write_shared_master(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="1" * 16)
    size = (tmp_path / "a.fits").stat().st_size
    assert sidecar_is_current(tmp_path / "a.fits", size), "published by AstroFlow: the size is enough"
    assert not sidecar_is_current(tmp_path / "a.fits", size + 100)
    (tmp_path / "b.fits").write_bytes(b"x" * 10)
    assert not sidecar_is_current(tmp_path / "b.fits", 10), "no sidecar at all"
    # measured here: size AND mtime must match — a same-size re-push is new bytes
    st = (tmp_path / "a.fits").stat()
    set_section(tmp_path / "a.fits", MEASURED, {"size": st.st_size, "mtime_ns": st.st_mtime_ns})
    assert sidecar_is_current(tmp_path / "a.fits", size)
    os.utime(tmp_path / "a.fits", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert not sidecar_is_current(tmp_path / "a.fits", size)


def test_measuring_in_place_drops_a_stale_sidecar_first_and_records_the_bytes(tmp_path, monkeypatch):
    import metadata.image_metadata as im
    from file_io.sidecar import get_section
    from metadata.shared_masters import MEASURED
    data = tmp_path / "a.fits"
    write_shared_master(data, call_hash="a" * 16, stable_hash="1" * 16)
    with open(data, "ab") as f:
        f.write(b"\0" * 100)   # re-pushed, or still arriving, when the sidecar was written
    seen = []
    monkeypatch.setattr(im.compute_image_metadata, "_function",
                        lambda path: seen.append((path, (tmp_path / "a.fits.json").exists())))
    compute_shared_master._function(str(data), data.stat().st_size, 0)
    assert seen == [(data, False)], "the stale sidecar was gone before the file was measured"
    st = data.stat()
    assert get_section(data, MEASURED) == {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    # current now: measuring again is a no-op
    seen.clear()
    compute_shared_master._function(str(data), data.stat().st_size, 0)
    assert seen == []


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
