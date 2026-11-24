"""`refdata.fetch` against a store on disk: files arrive verified, a bad copy is
mended, the store stands in for a source that fails, and status counts."""
import hashlib
from pathlib import Path

import pytest

from refdata import fetch as rf


def _store(tmp_path: Path) -> tuple[Path, Path, str]:
    """Two files published by URL, one of them also in the store; a manifest naming them."""
    upstream, store = tmp_path / "upstream", tmp_path / "store"
    upstream.mkdir(), store.mkdir()
    a, b = b"index " * 1000, b"gaia " * 2000
    sha_a, sha_b = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
    (upstream / "index-4107.fits").write_bytes(a)
    (store / sha_b).write_bytes(b)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("# sha256\tbytes\tpath\turl\n"
                        f"{sha_a}\t{len(a)}\tastrometry-index/index-4107.fits\t{(upstream / 'index-4107.fits').as_uri()}\n"
                        f"{sha_b}\t{len(b)}\tgaia_xp/flux.npy\tstore\n")
    return manifest, store, store.as_uri()


def test_fetches_from_the_source_and_the_store(tmp_path):
    manifest, store, store_url = _store(tmp_path)
    root = tmp_path / "refdata"
    assert rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None) == 2
    assert (root / "astrometry-index" / "index-4107.fits").read_bytes() == b"index " * 1000
    assert (root / "gaia_xp" / "flux.npy").read_bytes() == b"gaia " * 2000
    assert rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None) == 0
    assert rf.status(root=root, manifest=manifest)["complete"]


def test_mends_a_file_that_differs(tmp_path):
    manifest, store, store_url = _store(tmp_path)
    root = tmp_path / "refdata"
    rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None)
    bad = root / "gaia_xp" / "flux.npy"
    bad.write_bytes(b"xxxx " * 2000)                       # same size, other bytes
    assert rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None) == 0
    assert rf.fetch(hashed=True, root=root, manifest=manifest, store=store_url, log=lambda s: None) == 1
    assert bad.read_bytes() == b"gaia " * 2000


def test_the_store_stands_in_for_a_failed_source(tmp_path):
    manifest, store, store_url = _store(tmp_path)
    root = tmp_path / "refdata"
    sha_a, size_a, path_a, url_a = rf.read_manifest(manifest)[0]
    Path(url_a.removeprefix("file://")).unlink()           # the source is gone
    with pytest.raises(SystemExit):
        rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None)
    (store / sha_a).write_bytes(b"index " * 1000)          # the store has a copy
    assert rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None) == 2


def test_rejects_a_source_whose_bytes_do_not_match(tmp_path):
    manifest, store, store_url = _store(tmp_path)
    root = tmp_path / "refdata"
    sha_a, size_a, path_a, url_a = rf.read_manifest(manifest)[0]
    Path(url_a.removeprefix("file://")).write_bytes(b"other " * 1000)
    with pytest.raises(SystemExit):
        rf.fetch(root=root, manifest=manifest, store=store_url, log=lambda s: None)
    assert not (root / path_a).exists() and not list((root / "astrometry-index").glob("*.part"))


def test_status_counts_by_folder(tmp_path):
    manifest, store, store_url = _store(tmp_path)
    s = rf.status(root=tmp_path / "refdata", manifest=manifest)
    assert not s["complete"]
    assert s["folders"]["astrometry-index"] == {"files": 1, "present": 0, "bytes": 6000, "missing_bytes": 6000}
    assert s["folders"]["gaia_xp"]["missing_bytes"] == 10000
