"""Recover or reap the staging files left behind by a killed publish.

Two publish protocols leave differently-shaped debris, and they need OPPOSITE
treatment — which is the whole reason this module exists rather than a `find
-delete` in the Makefile.

**Parked payloads** — `products/registry._publish_with_presentations` renames
the payload ASIDE (`os.replace(source, staging)`) so the pyramid can be built
before the payload lands, then republishes in a `finally`. The `finally` covers
exceptions but NOT a killed process, and its own comment names the consequence:
"the payload is stranded under its dot-name and the product vanishes entirely".
So a leftover `.{stem}.staging.{pid}{suffix}` whose canonical name is MISSING is
not garbage — it IS the product, and the correct action is to RESTORE it.

    (learned the hard way on 2026-08-17: an 18.9 GB parked
    `combine_broadband_rgb` payload was deleted as "obvious litter" during a
    disk emergency. Its canonical file never existed again.)

**Partial copies** — `file_io.publish.publish_atomically` copies to
`{dest}.tmp.{pid}.{uuid}` and then replaces. The destination is untouched until
the replace, so a leftover `.tmp.` file is only ever a half-written copy: always
safe to delete, never something to restore.

SAFETY. Both actions move or unlink files another process might be writing, so
every candidate must satisfy BOTH guards:

  * the writing PID is no longer alive, and
  * the file has not been modified for `min_age_s`.

The PID check alone is not enough on a multi-node cluster sharing a filesystem:
PIDs are per-machine, so a live writer on the MBP can look dead here. The age
guard is what actually makes that safe, which is why the intended invocation is
from `make start`, with the cluster down. A live PID always means SKIP — erring
toward keeping a file we cannot vouch for.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# `.{stem}.staging.{pid}{suffix}` — registry.py's parked payload.
_PARKED = re.compile(r"^\.(?P<stem>.+)\.staging\.(?P<pid>\d+)(?P<suffix>\.[^.]+)$")
# `{dest.name}.tmp.{pid}.{uuid}` — publish.py's in-flight copy.
_PARTIAL = re.compile(r"^(?P<dest>.+)\.tmp\.(?P<pid>\d+)\.[0-9a-f]+$")

RESTORE, DELETE, SKIP = "restore", "delete", "skip"

# One hour: longer than the gap between a park and its republish (header stamp
# + pyramid build), short enough that debris from a crashed run is collected on
# the next start rather than lingering for days.
DEFAULT_MIN_AGE_S = 3600.0


@dataclass(frozen=True)
class Candidate:
    path: Path
    action: str
    reason: str
    size: int
    target: "Path | None" = None   # where a RESTORE would land


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        # No psutil: assume alive. Refusing to act is always the safe default.
        return True


def scan(roots, min_age_s: float = DEFAULT_MIN_AGE_S, now: float | None = None
         ) -> "list[Candidate]":
    """Classify every staging file under `roots` without touching anything."""
    import time
    now = time.time() if now is None else now
    out: list[Candidate] = []

    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(d.iterdir()):
                if not f.is_file():
                    continue
                m = _PARKED.match(f.name)
                partial = None if m else _PARTIAL.match(f.name)
                if not m and not partial:
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                pid = int((m or partial).group("pid"))
                age = now - st.st_mtime

                if _alive(pid):
                    out.append(Candidate(f, SKIP, f"writer pid {pid} alive",
                                         st.st_size))
                    continue
                if age < min_age_s:
                    out.append(Candidate(
                        f, SKIP, f"only {age / 60:.0f} min old", st.st_size))
                    continue

                if partial:
                    out.append(Candidate(
                        f, DELETE, "partial copy; destination untouched",
                        st.st_size))
                    continue

                canonical = f.with_name(m.group("stem") + m.group("suffix"))
                if canonical.exists():
                    out.append(Candidate(
                        f, DELETE, "canonical present; stale duplicate",
                        st.st_size, canonical))
                else:
                    out.append(Candidate(
                        f, RESTORE, "canonical MISSING; this is the payload",
                        st.st_size, canonical))
    return out


def sweep(roots, min_age_s: float = DEFAULT_MIN_AGE_S, dry_run: bool = True
          ) -> dict:
    """Apply `scan`'s verdicts. Returns a report; `dry_run` decides nothing."""
    cands = scan(roots, min_age_s=min_age_s)
    report = {"restored": 0, "deleted": 0, "skipped": 0,
              "restored_bytes": 0, "deleted_bytes": 0,
              "candidates": cands, "dry_run": dry_run}
    for c in cands:
        if c.action == SKIP:
            report["skipped"] += 1
            continue
        if dry_run:
            report["restored" if c.action == RESTORE else "deleted"] += 1
            report[("restored_bytes" if c.action == RESTORE
                    else "deleted_bytes")] += c.size
            continue
        try:
            if c.action == RESTORE:
                # Re-check under the race: if the canonical appeared since the
                # scan, this parked copy is now redundant rather than the only
                # copy, and restoring would clobber the newer file.
                if c.target is not None and not c.target.exists():
                    os.replace(c.path, c.target)
                    report["restored"] += 1
                    report["restored_bytes"] += c.size
                else:
                    report["skipped"] += 1
            else:
                os.unlink(c.path)
                report["deleted"] += 1
                report["deleted_bytes"] += c.size
        except OSError:
            report["skipped"] += 1
    return report
