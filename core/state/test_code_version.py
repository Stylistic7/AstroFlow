"""GitOps pull-deploy foundation: committed + working-tree versions/snapshots,
extraction, and the ASTROFLOW_LIVE_CORE base-path override. Ray-free; uses a
throwaway git repo.
"""
import subprocess
from pathlib import Path

import pytest

from state.code_version import (
    SYNCED_SHA_FILE,
    build_snapshot_tar,
    build_working_tree_tar,
    deployed_code_version,
    head_cluster_version,
    is_deployable_version,
    read_synced_sha,
    tar_directory,
    verify_and_extract,
    working_tree_version,
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _head_sha(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@t.t")
    _git(r, "config", "user.name", "t")
    (r / "core").mkdir()
    (r / "core" / "main.py").write_text("print('v1')\n")
    (r / ".gitignore").write_text("ignored.txt\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "v1")
    return r


# ---- committed versions -----------------------------------------------------

def test_committed_version_is_head_sha(repo):
    v = head_cluster_version(repo, uncommitted=False)
    assert v == _head_sha(repo)
    assert is_deployable_version(v)


def test_committed_mode_ignores_working_tree_edits(repo):
    # Default mode always deploys HEAD, even with uncommitted edits present.
    sha = _head_sha(repo)
    (repo / "core" / "main.py").write_text("print('EDIT')\n")
    assert head_cluster_version(repo, uncommitted=False) == sha


def test_committed_snapshot_excludes_uncommitted_changes(repo, tmp_path):
    sha = _head_sha(repo)
    (repo / "core" / "main.py").write_text("print('EDIT')\n")
    (repo / "core" / "scratch.py").write_text("new\n")
    dest = tmp_path / "checkout"
    verify_and_extract(build_snapshot_tar(repo, sha), sha, dest)
    assert (dest / "core" / "main.py").read_text() == "print('v1')\n"
    assert not (dest / "core" / "scratch.py").exists()


# ---- uncommitted (working-tree) versions ------------------------------------

def test_working_tree_version_is_deployable_and_changes_on_edit(repo):
    v1 = head_cluster_version(repo, uncommitted=True)
    assert v1.startswith("wt:") and is_deployable_version(v1)
    (repo / "core" / "main.py").write_text("print('EDITED')\n")
    v2 = working_tree_version(repo)
    assert v2 != v1 and is_deployable_version(v2)


def test_working_tree_snapshot_includes_uncommitted_and_untracked(repo, tmp_path):
    (repo / "core" / "main.py").write_text("print('EDITED')\n")  # modified, uncommitted
    (repo / "core" / "brand_new.py").write_text("x = 1\n")        # untracked, not ignored
    (repo / "ignored.txt").write_text("secret\n")                 # gitignored

    v = working_tree_version(repo)
    dest = tmp_path / "checkout"
    verify_and_extract(build_working_tree_tar(repo), v, dest)

    assert (dest / "core" / "main.py").read_text() == "print('EDITED')\n"
    assert (dest / "core" / "brand_new.py").read_text() == "x = 1\n"
    assert not (dest / "ignored.txt").exists()       # ignored stays put
    assert read_synced_sha(dest) == v


def test_excludes_are_dropped_from_working_tree(repo):
    (repo / "testdata").mkdir()
    (repo / "testdata" / "big.bin").write_text("blob\n")
    from state.code_version import _working_tree_files
    assert not any(f.startswith("testdata/") for f in _working_tree_files(repo))


# ---- is_deployable_version --------------------------------------------------

def test_is_deployable_version_rules():
    assert is_deployable_version("a" * 40)
    assert is_deployable_version("wt:" + "a" * 40)
    assert not is_deployable_version("A" * 40)        # uppercase isn't git hex
    assert not is_deployable_version("wt:" + "a" * 39)
    assert not is_deployable_version("a" * 39)
    assert not is_deployable_version("")
    assert not is_deployable_version("dev")


def test_deployed_code_version_reads_frozen_sha(tmp_path, monkeypatch):
    """Reads the SHA the running app froze into its checkout — the exact code
    version driving the trace/tap-snapshot cache keys."""
    (tmp_path / SYNCED_SHA_FILE).write_text("a" * 40)
    monkeypatch.setattr("base_path.checkout_dir", lambda: tmp_path)
    assert deployed_code_version() == "a" * 40


def test_deployed_code_version_empty_when_no_frozen_checkout(tmp_path, monkeypatch):
    """No frozen SHA file (dev shells, tests) → "" so trace keys degrade to the
    prior pipeline-source-only behavior rather than raising."""
    monkeypatch.setattr("base_path.checkout_dir", lambda: tmp_path)  # empty dir
    assert deployed_code_version() == ""


def test_deployed_code_version_swallows_errors(monkeypatch):
    """An unconfigured base path must not blow up a trace — degrade to ""."""
    def boom():
        raise RuntimeError("base path not configured")
    monkeypatch.setattr("base_path.checkout_dir", boom)
    assert deployed_code_version() == ""


# ---- extraction / serving ---------------------------------------------------

def test_extract_replaces_existing_checkout(repo, tmp_path):
    sha1 = _head_sha(repo)
    dest = tmp_path / "checkout"
    verify_and_extract(build_snapshot_tar(repo, sha1), sha1, dest)
    (dest / "core" / "stale.py").write_text("old\n")

    (repo / "core" / "main.py").write_text("print('v2')\n")
    _git(repo, "commit", "-aqm", "v2")
    sha2 = _head_sha(repo)
    verify_and_extract(build_snapshot_tar(repo, sha2), sha2, dest)

    assert (dest / "core" / "main.py").read_text() == "print('v2')\n"
    assert not (dest / "core" / "stale.py").exists()
    assert read_synced_sha(dest) == sha2


def test_corrupt_tar_leaves_dest_untouched(repo, tmp_path):
    sha = _head_sha(repo)
    dest = tmp_path / "checkout"
    verify_and_extract(build_snapshot_tar(repo, sha), sha, dest)
    with pytest.raises(Exception):
        verify_and_extract(b"not a gzip", "deadbeef", dest)
    assert (dest / "core" / "main.py").read_text() == "print('v1')\n"
    assert read_synced_sha(dest) == sha


def test_tar_directory_roundtrips_and_drops_sidecar(repo, tmp_path):
    # How the head serves the frozen checkout it runs: tar it, re-extract, same
    # content, and the version sidecar is NOT shipped (worker rewrites its own).
    sha = _head_sha(repo)
    src = tmp_path / "checkout"
    verify_and_extract(build_snapshot_tar(repo, sha), sha, src)

    served = tar_directory(src)
    dest = tmp_path / "worker"
    verify_and_extract(served, sha, dest)
    assert (dest / "core" / "main.py").read_text() == "print('v1')\n"

    import tarfile, io
    with tarfile.open(fileobj=io.BytesIO(served), mode="r:gz") as tf:
        assert SYNCED_SHA_FILE not in tf.getnames()


def test_base_path_honors_live_core_override(monkeypatch, tmp_path):
    import base_path
    monkeypatch.setattr(base_path, "live_core_pointer_file",
                        lambda: tmp_path / "no_pointer")
    checkout = tmp_path / "checkout"
    (checkout / "core").mkdir(parents=True)
    monkeypatch.setenv("ASTROFLOW_LIVE_CORE", str(checkout))
    assert str(base_path.get_base_path()) == str(checkout)
    monkeypatch.setenv("ASTROFLOW_LIVE_CORE", str(tmp_path / "gone"))
    assert base_path.get_base_path() == Path(base_path.__file__).resolve().parent.parent
    monkeypatch.delenv("ASTROFLOW_LIVE_CORE")
    assert base_path.get_base_path() == Path(base_path.__file__).resolve().parent.parent
