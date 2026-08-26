"""Code-version identity for the cluster. Code moves via GIT, not via the app.

The cluster must run ONE version of the software. The head advertises the
version it booted with; a worker computes its own version the same way and
REFUSES to join on a mismatch (`state_machine._join_head`). The remedy is
`git pull` on the worker, after which it joins on the next discovery tick.

This replaced a pull-deploy snapshot system (the head froze a checkout under
Application Support, served it over HTTP, and workers extracted + restarted
into it). That was a second, parallel code-distribution mechanism with its
own failure modes, and on 2026-08-25 it failed exactly like one: a stale
frozen checkout kept the ADVERTISED version current while Ray tasks —
pointed at the frozen dir via `working_dir` — ran old code on the head and
on every worker that pulled the snapshot. The symptom was
`ModuleNotFoundError` deep inside placements, poisoning whole subtrees while
every health signal looked green. One mechanism now: git.

Versions:
  * committed (default): the repo's HEAD SHA. A worker must be at the SAME
    SHA with a CLEAN tree (excludes aside) — edited code under a committed
    banner is a lie, so a dirty tree reports `<sha>-dirty` and can't match.
  * uncommitted (`ASTROFLOW_UNCOMMITTED=1` / persisted setting): `wt:<hash>`
    over the working tree's deployable files. A worker only matches when its
    tree is byte-identical — practically a single-machine/dev mode, and a
    worker refuses loudly, which is the point.
"""
import hashlib
import subprocess
from pathlib import Path

# Single source of truth for the file set shipped to Ray task workers via
# `runtime_env.working_dir`, ignored by the working-tree hash, and ignored
# when deciding whether a tree is dirty.
RAY_WORKING_DIR_EXCLUDES = [
    ".cache/",
    "cache.tar.gz",
    ".git/",
    "web_server/frontend/",
    "web_server/backend/static/",
    "testdata/",
    "pix/testdata/",
]

_WT_PREFIX = "wt:"
_HEX = set("0123456789abcdef")


def is_deployable_version(version: str) -> bool:
    """A worker fences against a version only if it names concrete code: a
    committed SHA (40 hex) or a working-tree hash (`wt:<40 hex>`). Empty /
    other markers are non-deployable — the worker waits."""
    h = version[len(_WT_PREFIX):] if version.startswith(_WT_PREFIX) else version
    return len(h) == 40 and all(c in _HEX for c in h)


def _is_excluded(rel: str) -> bool:
    """Whether a repo-root-relative path is excluded from version identity.

    `RAY_WORKING_DIR_EXCLUDES` is relative to `core/` (Ray's working_dir), but
    git lists paths from the repo ROOT (e.g. `core/testdata/x`). Match the path
    AND the path with a leading `core/` stripped."""
    for e in RAY_WORKING_DIR_EXCLUDES:
        name = e.rstrip("/")
        for candidate in (rel, rel[5:] if rel.startswith("core/") else rel):
            if candidate == name or candidate.startswith(name + "/"):
                return True
    return False


def _head_sha(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _working_tree_files(repo: Path) -> list[str]:
    """Tracked + untracked-but-not-ignored files, minus the excludes —
    i.e. exactly the files that constitute the deployable working tree."""
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-co", "--exclude-standard"],
        capture_output=True, text=True, check=True,
    )
    return sorted(
        f for f in out.stdout.splitlines() if f and not _is_excluded(f)
    )


def working_tree_version(repo: Path) -> str:
    """`wt:<hash>` over the working tree's deployable files (path + content),
    so any save changes it."""
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


def _is_dirty(repo: Path) -> bool:
    """Whether the tree differs from HEAD in any non-excluded file."""
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    for line in out.stdout.splitlines():
        # Porcelain: `XY <path>`; renames render as `old -> new`.
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if path and not _is_excluded(path):
            return True
    return False


def local_code_version(repo: Path, head_version: str) -> str:
    """THIS machine's version, computed in the head's mode so the two are
    comparable: `wt:` head -> our working-tree hash; committed head -> our
    HEAD SHA, suffixed `-dirty` when the tree has non-excluded edits so
    edited code can never pass the fence under a committed banner."""
    repo = Path(repo)
    if head_version.startswith(_WT_PREFIX):
        return working_tree_version(repo)
    sha = _head_sha(repo)
    return f"{sha}-dirty" if _is_dirty(repo) else sha


def _unpushed_count(repo: Path) -> "int | None":
    """Commits on HEAD not reachable from the current branch's upstream, read
    from LOCAL refs (`@{u}..HEAD`) — no network, no fetch: `git push` updates
    the local upstream ref, so "pushed" is decidable offline. None when no
    upstream is configured (bare dev/test repos, detached HEAD)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "@{u}..HEAD"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return None
    return int(out.stdout.strip())


def require_clean_and_pushed(repo: Path) -> tuple[bool, str]:
    """The head's start gate: `(ok, reason)`. The cluster's version is the
    HEAD SHA, so the head must be running exactly that (clean tree) and the
    workers must be able to `git pull` it (pushed). Refusing HERE is what
    makes the deploy flow un-forgettable: commit -> push -> start the head ->
    pull on workers -> start workers."""
    if _is_dirty(repo):
        return False, "the working tree has uncommitted changes"
    n = _unpushed_count(repo)
    if n is None:
        return True, "no upstream configured; skipping the pushed check"
    if n:
        return False, f"{n} commit(s) not pushed to the upstream"
    return True, ""


_deployed_version_cache: "str | None" = None


def deployed_code_version() -> str:
    """The code version this process is executing — folded into the pipeline
    trace / tap-snapshot cache keys so that editing ANY op or helper a
    pipeline calls re-derives the trace (see
    `trace_paths.pipeline_source_version`).

    Always the WORKING-TREE hash: it changes on any deployable edit (a strict
    superset of committed-SHA invalidation) and is identical across the
    cluster whenever the join fence holds (equal trees hash equal). Computed
    once per process; a mid-run edit re-keys only processes spawned after it —
    which is the mixed-code hazard the fence exists to make loud, not
    something to paper over.

    Returns "" when the checkout isn't a git repo (bundles, bare test dirs) —
    trace keys then simply lose their code-version component, as before.
    """
    global _deployed_version_cache
    if _deployed_version_cache is None:
        try:
            from base_path import get_base_path
            _deployed_version_cache = working_tree_version(
                Path(get_base_path()))
        except Exception:
            _deployed_version_cache = ""
    return _deployed_version_cache
