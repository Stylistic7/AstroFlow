"""Code-version identity: committed + working-tree versions, the worker-side
`local_code_version` fence input, and the per-process `deployed_code_version`
cache-key component. Ray-free; uses a throwaway git repo.
"""
import subprocess
from pathlib import Path

import pytest

import state.code_version as cv
from state.code_version import (
    deployed_code_version,
    head_cluster_version,
    is_deployable_version,
    local_code_version,
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
    assert head_cluster_version(repo, uncommitted=False) == _head_sha(repo)
    assert is_deployable_version(head_cluster_version(repo, False))


def test_committed_mode_ignores_working_tree_edits(repo):
    before = head_cluster_version(repo, uncommitted=False)
    (repo / "core" / "main.py").write_text("print('edited')\n")
    assert head_cluster_version(repo, uncommitted=False) == before


# ---- working-tree versions --------------------------------------------------

def test_working_tree_version_is_deployable_and_changes_on_edit(repo):
    v1 = working_tree_version(repo)
    assert v1.startswith("wt:") and is_deployable_version(v1)
    (repo / "core" / "main.py").write_text("print('edited')\n")
    v2 = working_tree_version(repo)
    assert v2 != v1


def test_working_tree_version_covers_untracked_but_not_ignored(repo):
    v1 = working_tree_version(repo)
    (repo / "core" / "new_module.py").write_text("x = 1\n")   # untracked
    v2 = working_tree_version(repo)
    assert v2 != v1, "a new untracked file must change the version"
    (repo / "ignored.txt").write_text("noise\n")              # gitignored
    assert working_tree_version(repo) == v2


def test_excludes_are_dropped_from_working_tree(repo):
    (repo / "core" / "testdata").mkdir()
    v1 = working_tree_version(repo)
    (repo / "core" / "testdata" / "big.bin").write_bytes(b"x" * 10)
    assert working_tree_version(repo) == v1


def test_is_deployable_version_rules():
    assert is_deployable_version("a" * 40)
    assert is_deployable_version("wt:" + "b" * 40)
    assert not is_deployable_version("")
    assert not is_deployable_version("dev")
    assert not is_deployable_version("a" * 39)
    assert not is_deployable_version(("a" * 39) + "-dirty")


# ---- local_code_version: the worker's side of the fence ---------------------

def test_local_version_matches_committed_head(repo):
    sha = _head_sha(repo)
    assert local_code_version(repo, sha) == sha


def test_local_version_reports_dirty_for_edited_tree(repo):
    """Edited code can never pass the fence under a committed banner."""
    sha = _head_sha(repo)
    (repo / "core" / "main.py").write_text("print('edited')\n")
    assert local_code_version(repo, sha) == f"{sha}-dirty"


def test_local_version_ignores_excluded_dirt(repo):
    """A change confined to the excludes (caches, testdata) is not dirt —
    those files are never part of the deployed identity."""
    sha = _head_sha(repo)
    (repo / "core" / "testdata").mkdir()
    (repo / "core" / "testdata" / "scratch.bin").write_bytes(b"x")
    assert local_code_version(repo, sha) == sha


def test_local_version_follows_the_heads_mode(repo):
    """A `wt:` head compares working trees; identical trees match even when
    HEAD SHAs would (trivially here) also match — the MODE comes from the
    head's advertised version, so the two sides always hash the same thing."""
    assert local_code_version(repo, "wt:" + "0" * 40) == working_tree_version(repo)


# ---- deployed_code_version: the trace/tap cache-key component ---------------

def test_deployed_code_version_is_the_working_tree_hash(repo, monkeypatch):
    import base_path
    monkeypatch.setattr(base_path, "get_base_path", lambda: str(repo))
    monkeypatch.setattr(cv, "_deployed_version_cache", None)
    assert deployed_code_version() == working_tree_version(repo)


def test_deployed_code_version_caches_per_process(repo, monkeypatch):
    import base_path
    monkeypatch.setattr(base_path, "get_base_path", lambda: str(repo))
    monkeypatch.setattr(cv, "_deployed_version_cache", None)
    first = deployed_code_version()
    (repo / "core" / "main.py").write_text("print('edited')\n")
    assert deployed_code_version() == first, (
        "cached per process — a mid-run edit re-keys only NEW processes")


def test_deployed_code_version_empty_outside_a_repo(tmp_path, monkeypatch):
    import base_path
    monkeypatch.setattr(base_path, "get_base_path", lambda: str(tmp_path))
    monkeypatch.setattr(cv, "_deployed_version_cache", None)
    assert deployed_code_version() == ""


# ---- require_clean_and_pushed: the head's start gate ------------------------

@pytest.fixture
def pushed_repo(repo, tmp_path):
    """`repo` with a bare upstream it has fully pushed to."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "-u", "origin", "HEAD")
    return repo


def test_gate_passes_clean_and_pushed(pushed_repo):
    ok, reason = cv.require_clean_and_pushed(pushed_repo)
    assert ok and reason == ""


def test_gate_refuses_a_dirty_tree(pushed_repo):
    (pushed_repo / "core" / "main.py").write_text("print('edited')\n")
    ok, reason = cv.require_clean_and_pushed(pushed_repo)
    assert not ok and "uncommitted" in reason


def test_gate_refuses_unpushed_commits(pushed_repo):
    (pushed_repo / "core" / "main.py").write_text("print('v2')\n")
    _git(pushed_repo, "commit", "-qam", "v2")
    ok, reason = cv.require_clean_and_pushed(pushed_repo)
    assert not ok and "not pushed" in reason


def test_gate_ignores_excluded_dirt(pushed_repo):
    (pushed_repo / "core" / "testdata").mkdir()
    (pushed_repo / "core" / "testdata" / "x.bin").write_bytes(b"x")
    ok, _ = cv.require_clean_and_pushed(pushed_repo)
    assert ok


def test_gate_skips_pushed_check_without_upstream(repo):
    """Bare dev/test repos have no upstream — clean is enough, and the reason
    says the check was skipped rather than silently passing."""
    ok, reason = cv.require_clean_and_pushed(repo)
    assert ok and "no upstream" in reason
