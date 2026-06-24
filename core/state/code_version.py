"""Code-version + snapshot for GitOps-style pull-deploy.

The cluster runs one **frozen snapshot** at a time, identified by a **version**:

  * committed (default): the version is the repo's HEAD SHA, and the snapshot is
    `git archive HEAD` — reproducible, ignores the working tree.
  * uncommitted (opt-in, `--uncommitted`): the version is `wt:<hash>` over the
    working tree, and the snapshot is a tar of that working tree (tracked +
    untracked-not-ignored, minus the excludes). Lets you deploy edits across the
    whole cluster without committing.

Either way the head freezes a snapshot, runs it, and ships it; workers pull the
same bytes. So the whole cluster runs identical software, committed or not.
Uncommitted runs simply aren't reproducible from git — which is why it's opt-in.

This module is the single source of truth for the snapshot file set and how
snapshots are built / extracted / versioned, so head and workers can't disagree.
"""
import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

# Single source of truth for the file set shipped to Ray task workers via
# `runtime_env.working_dir` AND excluded from working-tree snapshots.
RAY_WORKING_DIR_EXCLUDES = [
    ".cache/",
    "cache.tar.gz",
    ".git/",
    "web_server/frontend/",
    "web_server/backend/static/",
    "testdata/",
    "pix/testdata/",
]

# Sidecar written into a synced/frozen checkout recording its version. Read to
# decide whether a re-sync is needed and as the restart loop-breaker.
SYNCED_SHA_FILE = ".astroflow_synced_sha"

_WT_PREFIX = "wt:"
_HEX = set("0123456789abcdef")


def is_deployable_version(version: str) -> bool:
    """A node follows a version only if it names a concrete frozen snapshot: a
    committed SHA (40 hex) or a working-tree hash (`wt:<40 hex>`). Empty / other
    markers are non-deployable."""
    h = version[len(_WT_PREFIX):] if version.startswith(_WT_PREFIX) else version
    return len(h) == 40 and all(c in _HEX for c in h)


def _is_excluded(rel: str) -> bool:
    """Whether a repo-root-relative path is a snapshot exclude.

    `RAY_WORKING_DIR_EXCLUDES` is relative to `core/` (Ray's working_dir), but
    git lists paths from the repo ROOT (e.g. `core/testdata/x`). Match the path
    AND the path with a leading `core/` stripped, so `core/testdata/` (13 GB!)
    is dropped, not gzipped into the deploy snapshot."""
    candidates = [rel]
    if rel.startswith("core/"):
        candidates.append(rel[len("core/"):])
    return any(
        c == e.rstrip("/") or c.startswith(e.rstrip("/") + "/")
        for e in RAY_WORKING_DIR_EXCLUDES
        for c in candidates
    )


def _head_sha(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _working_tree_files(repo: Path) -> list[str]:
    """Tracked + untracked-but-not-ignored files, minus the snapshot excludes —
    i.e. exactly what git would consider deployable from the working tree."""
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    )
    return sorted(
        f for f in out.stdout.splitlines() if f and not _is_excluded(f)
    )


def working_tree_version(repo: Path) -> str:
    """`wt:<hash>` over the working tree's deployable files (path + content),
    so any save changes it and workers re-sync."""
    h = hashlib.sha256()
    for rel in _working_tree_files(repo):
        p = Path(repo) / rel
        if not p.is_file():
            continue
        h.update(rel.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return _WT_PREFIX + h.hexdigest()[:40]


def head_cluster_version(repo: Path, uncommitted: bool) -> str:
    """The version this head advertises: a `wt:` working-tree hash in
    uncommitted mode, else the committed HEAD SHA."""
    return working_tree_version(repo) if uncommitted else _head_sha(repo)


def build_working_tree_tar(repo: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in _working_tree_files(repo):
            p = Path(repo) / rel
            if p.is_file():
                tf.add(p, arcname=rel)
    return buf.getvalue()


def build_snapshot_tar(repo: Path, version: str) -> bytes:
    """The frozen snapshot for `version`: the working tree for a `wt:` version,
    else `git archive <sha>` (committed, reproducible). Both drop the same heavy
    excludes so the two snapshot kinds — and what Ray ships — stay consistent."""
    if version.startswith(_WT_PREFIX):
        return build_working_tree_tar(repo)
    # git pathspecs: drop the excluded dirs (".git/" never archives). The
    # excludes are core/-relative, but archive runs from the repo root, so
    # exclude BOTH the bare path and the core/-prefixed path (covers
    # core/testdata/ etc., the 13 GB we must not ship).
    exclude_specs = []
    for e in RAY_WORKING_DIR_EXCLUDES:
        name = e.rstrip("/")
        if name == ".git":
            continue
        exclude_specs.append(f":(exclude){name}")
        exclude_specs.append(f":(exclude)core/{name}")
    out = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar.gz", version,
         "--", ".", *exclude_specs],
        capture_output=True, check=True,
    )
    return out.stdout


def tar_directory(path: Path) -> bytes:
    """Tar an extracted checkout (minus the version sidecar) — how the head
    serves the exact frozen snapshot it is running."""
    path = Path(path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.name != SYNCED_SHA_FILE:
                tf.add(p, arcname=str(p.relative_to(path)))
    return buf.getvalue()


def verify_and_extract(tar_bytes: bytes, version: str, dest: Path) -> None:
    """Atomically replace `dest` with the extracted snapshot, recording
    `version`. Extraction is the integrity check; the caller re-confirms the
    head's advertised version just before, closing the race window."""
    dest = Path(dest)
    staging = dest.with_name(dest.name + ".new")
    backup = dest.with_name(dest.name + ".old")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        tf.extractall(staging, filter="data")

    (staging / SYNCED_SHA_FILE).write_text(version)

    if backup.exists():
        shutil.rmtree(backup)
    if dest.exists():
        os.rename(dest, backup)
    os.rename(staging, dest)
    if backup.exists():
        shutil.rmtree(backup)


def read_synced_sha(dest: Path) -> str | None:
    """The version recorded in a synced/frozen checkout, or None."""
    f = Path(dest) / SYNCED_SHA_FILE
    return f.read_text().strip() if f.exists() else None
