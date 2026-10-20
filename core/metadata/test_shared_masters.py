"""Reading a shared master folder (`metadata.shared_masters`): the scan
lists masters by their sidecars and leaves a sync tool's temporaries out;
a master is admissible only when whole, stamped, solved and a light."""
from pathlib import Path

from metadata.shared_masters import admissible_masters, scan_shared_dir
from testing.shared_masters import solved_master_metadata, write_shared_master


def test_scan_lists_data_files_that_have_a_sidecar_and_skips_dot_files(tmp_path):
    write_shared_master(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="1" * 16)
    (tmp_path / "loose.fits").write_bytes(b"no sidecar")
    (tmp_path / ".syncthing.b.fits.tmp").write_bytes(b"arriving")
    (tmp_path / ".syncthing.b.fits.tmp.json").write_text("{}")
    (tmp_path / "notes.json").write_text("{}")
    record = scan_shared_dir(tmp_path)
    assert [rel for rel, _, _ in record.listing] == ["a.fits"]
    assert record.fingerprint == scan_shared_dir(tmp_path).fingerprint


def test_the_fingerprint_moves_when_a_sidecar_changes(tmp_path):
    write_shared_master(tmp_path / "a.fits", call_hash="a" * 16, stable_hash="1" * 16)
    before = scan_shared_dir(tmp_path).fingerprint
    from file_io.sidecar import set_section
    import os, time
    set_section(tmp_path / "a.fits", "shared", {"observer": "W"})
    os.utime(tmp_path / "a.fits.json", (time.time() + 5, time.time() + 5))
    assert scan_shared_dir(tmp_path).fingerprint != before


def test_only_whole_stamped_solved_light_masters_are_admissible(tmp_path):
    good = write_shared_master(tmp_path / "good.fits", call_hash="a" * 16, stable_hash="1" * 16)
    unstamped = solved_master_metadata(tmp_path / "unstamped.fits", call_hash="b" * 16, stable_hash="2" * 16)
    del unstamped.header["PIPEHASH"]
    write_shared_master(tmp_path / "unstamped.fits", unstamped)
    flat = solved_master_metadata(tmp_path / "flat.fits", call_hash="c" * 16, stable_hash="3" * 16)
    flat.header["IMAGETYP"] = "master flat"
    write_shared_master(tmp_path / "flat.fits", flat)
    write_shared_master(tmp_path / "partial.fits", call_hash="d" * 16, stable_hash="4" * 16)
    with open(tmp_path / "partial.fits", "ab") as f:
        f.write(b"\0" * 100)   # the sync tool is still delivering it

    listing = scan_shared_dir(tmp_path).listing
    masters = admissible_masters(tmp_path, listing)
    assert set(masters) == {good.content_hash}
    assert masters[good.content_hash].plate_solution is not None
    assert masters[good.content_hash].filename == tmp_path / "good.fits"
