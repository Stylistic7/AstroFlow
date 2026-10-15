"""Reap the scratch files a killed producer leaves behind.

Every image producer writes to a SCRATCH name beside its product —
`.{stem}.partial.{pid}.{tag}{suffix}`, plus `.pyr` for a pyramid built from it
(`products.registry.makes_image_wrap`) — and the product appears only through
one final `os.replace`. So a leftover scratch file is never the product, only
an unfinished one, and is always safe to delete. (Until 2026-10-15 the
payload was parked aside under a `.staging.` name while its pyramid was
built, and a stranded one WAS the product: an 18.9 GB master was lost to a
"litter" cleanup on 2026-08-17. The scratch protocol removed that case.)

`file_io.publish.publish_atomically` leaves `{dest}.tmp.{pid}.{uuid}` copies
the same way, with the destination untouched.

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

# `.{stem}.partial.{pid}.{tag}{suffix}[.pyr]` — makes_image_wrap's scratch.
_SCRATCH = re.compile(r"^\..+\.partial\.(?P<pid>\d+)\.[0-9a-f]+\.[^.]+(\.pyr)?$")
# `{dest.name}.tmp.{pid}.{uuid}` — publish.py's in-flight copy.
_PARTIAL = re.compile(r"^(?P<dest>.+)\.tmp\.(?P<pid>\d+)\.[0-9a-f]+$")

DELETE, SKIP = "delete", "skip"

# One hour: longer than any write-then-publish, short enough that debris from
# a crashed run is collected on the next start rather than lingering for days.
DEFAULT_MIN_AGE_S = 3600.0


@dataclass(frozen=True)
class Candidate:
    path: Path
    action: str
    reason: str
    size: int


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        # No psutil: assume alive. Refusing to act is always the safe default.
        return True


def scan(roots, min_age_s: float = DEFAULT_MIN_AGE_S, now: float | None = None
         ) -> "list[Candidate]":
    """Classify every scratch file under `roots` without touching anything."""
    import time
    now = time.time() if now is None else now
    out: list[Candidate] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(d.iterdir()):
                m = _SCRATCH.match(f.name) or _PARTIAL.match(f.name)
                if m is None or not f.is_file():
                    continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                pid = int(m.group("pid"))
                if _alive(pid):
                    out.append(Candidate(f, SKIP, f"writer pid {pid} alive", st.st_size))
                elif now - st.st_mtime < min_age_s:
                    out.append(Candidate(f, SKIP, f"only {(now - st.st_mtime) / 60:.0f} min old",
                                         st.st_size))
                else:
                    out.append(Candidate(f, DELETE, "unfinished write; the product is untouched",
                                         st.st_size))
    return out


def sweep(roots, min_age_s: float = DEFAULT_MIN_AGE_S, dry_run: bool = True
          ) -> dict:
    """Apply `scan`'s verdicts. Returns a report; `dry_run` decides nothing."""
    cands = scan(roots, min_age_s=min_age_s)
    report = {"deleted": 0, "skipped": 0, "deleted_bytes": 0,
              "candidates": cands, "dry_run": dry_run}
    for c in cands:
        if c.action == SKIP:
            report["skipped"] += 1
            continue
        try:
            if not dry_run:
                os.unlink(c.path)
            report["deleted"] += 1
            report["deleted_bytes"] += c.size
        except OSError:
            report["skipped"] += 1
    return report
