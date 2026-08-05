"""Wall-clock janitor for abnormal-termination debris.

Processes that die ungracefully — scheduler cancellation of over-submitted
duplicates, ENOSPC crash cascades, the PixInsight inactivity watchdog,
`make stop` — strand their temp files, and nothing else ever reclaims them:
not an app restart, not a fresh instance of the same tool, not even a reboot
(the Darwin user Caches dir is never cleared). Left alone the debris
compounds: 79 GB of PixInsight swaps accumulated Jan 2024 → Jul 2026, and
310 GB of pyvips temps in TWO DAYS of churny gigapixel rebuilds (2026-08-04).

The failure mode is structural: cleanup keyed to *process lifecycle* fails
exactly when processes fail. The janitor is therefore keyed to WALL CLOCK and
FILES — every tick it deletes debris that is (a) matched by a registered
spec, (b) older than the spec's grace period, and (c) provably abandoned:
not open by any live process (lsof), plus any spec-specific guard. A live
op's temp is always open; an orphan is never reopened — so the janitor is
safe to run while the app is processing.

Adding a debris class = one `DebrisSpec` in `SPECS`. Keep this module free of
app imports (stdlib + structlog only): it must be runnable anywhere, anytime,
including standalone (`python -m cache.janitor` for a single tick).

Scope: the local machine only. Each cluster node strands debris on its own
boot volume; workers get their own janitor when the worker runtime grows one
(today the head — where StateActor lives — is where the churn concentrates).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import structlog

log = structlog.get_logger()

# One tick every 10 minutes: fast enough that a kill-storm self-heals long
# before it threatens the disk (worst observed leak rate ~150 GB/day), slow
# enough that the lsof scans are noise.
DEFAULT_INTERVAL_S = 600.0


def _darwin_dir(confstr_name: str) -> Path | None:
    """One of the per-user Darwin dirs (Caches `C/` or Temp `T/`)."""
    try:
        value = os.confstr(confstr_name)
    except (ValueError, OSError):
        return None
    return Path(value) if value else None


def _temp_dirs() -> list[Path]:
    return [d for d in [_darwin_dir("CS_DARWIN_USER_TEMP_DIR")] if d]


def _cache_and_temp_dirs() -> list[Path]:
    dirs = [
        _darwin_dir("CS_DARWIN_USER_CACHE_DIR"),
        _darwin_dir("CS_DARWIN_USER_TEMP_DIR"),
    ]
    return [d for d in dirs if d]


def _process_running(pattern: str) -> bool:
    """True if any live process command line matches `pattern` (pgrep -f)."""
    try:
        return (
            subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return True  # can't tell -> assume running -> delete nothing


def _open_paths(candidates: list[Path]) -> set[Path]:
    """The subset of `candidates` open by ANY live process.

    Directories are scanned with `+D` (a dir is "in use" when anything under
    it is open). On any lsof failure the answer is ALL candidates — the
    janitor's failure posture is always "delete nothing".
    """
    if not candidates:
        return set()
    # lsof reports RESOLVED names (/private/var/... where the Darwin dirs are
    # reached via the /var symlink) — canonicalize both sides or every open
    # file silently looks orphaned.
    real = {c: os.path.realpath(c) for c in candidates}
    files = [c for c in candidates if not c.is_dir()]
    dirs = [c for c in candidates if c.is_dir()]
    open_names: set[str] = set()
    try:
        if files:
            out = subprocess.run(
                ["lsof", "-Fn", "--", *(real[c] for c in files)],
                capture_output=True, text=True, timeout=60,
            )
            open_names |= {
                line[1:] for line in out.stdout.splitlines()
                if line.startswith("n")
            }
        for d in dirs:
            out = subprocess.run(
                ["lsof", "-Fn", "+D", real[d]],
                capture_output=True, text=True, timeout=60,
            )
            if any(line.startswith("n") for line in out.stdout.splitlines()):
                open_names.add(real[d])
    except Exception:
        log.warning("janitor lsof failed; treating all candidates as live")
        return set(candidates)
    return {c for c in candidates if real[c] in open_names}


def _newest_mtime(path: Path) -> float:
    """mtime of `path`, or for a dir the newest of it and its top-level
    entries — a live WBPP run keeps writing into subdirs, so the top dir's
    own mtime alone would look stale."""
    try:
        newest = path.stat().st_mtime
        if path.is_dir():
            for child in path.iterdir():
                newest = max(newest, child.stat().st_mtime)
        return newest
    except OSError:
        return time.time()  # racing a concurrent delete -> treat as fresh


def _looks_like_wbpp_staging(path: Path) -> bool:
    """A generic `tmp*` TemporaryDirectory is WBPP staging iff it contains
    the layout WBPP creates. Guard keyed on CONTENT, not name, so foreign
    tmp dirs in T/ are never touched."""
    try:
        names = {p.name for p in path.iterdir()}
    except OSError:
        return False
    return bool(
        {"calibrated", "master"} & names
        or any(n.startswith(("RUN_", "WBPP-LOG")) for n in names)
    )


@dataclass(frozen=True)
class DebrisSpec:
    """One class of reclaimable debris.

    A candidate is deleted only when ALL hold on the same tick:
      * name matches `pattern` in one of `dirs()`
      * `enabled()` (a spec-wide gate, e.g. "no PixInsight running")
      * `guard(path)` (per-path gate, e.g. "looks like WBPP staging")
      * older than `min_age_s` (see `_newest_mtime`)
      * not open by any live process (lsof)
    """
    name: str
    dirs: Callable[[], list[Path]]
    pattern: str
    min_age_s: float
    enabled: Callable[[], bool] = field(default=lambda: True)
    guard: Callable[[Path], bool] = field(default=lambda p: True)


SPECS: tuple[DebrisSpec, ...] = (
    # Gigapixel pyvips materializations (12-14 GB each). Deleted by libvips
    # on clean close; leaked by every killed worker. A live one is always
    # open, so a short grace + lsof is sufficient.
    DebrisSpec(
        name="vips_temp",
        dirs=_temp_dirs,
        pattern="vips-*.v",
        min_age_s=15 * 60,
    ),
    # PixInsight swap files. PI deletes them only on graceful quit; the
    # watchdog / make stop strand them. PI holds swaps without keeping fds
    # open, so lsof can't attribute them — the sound rule is "no PixInsight
    # process at all", same as the make-start sweep.
    DebrisSpec(
        name="pixinsight_swap",
        dirs=_cache_and_temp_dirs,
        pattern="~PI~*.swp",
        min_age_s=10 * 60,
        enabled=lambda: not _process_running("PixInsight.app"),
    ),
    # stack_files' WBPP staging (generic tmp* TemporaryDirectory, ~46 GB for
    # a big re-stack). Content-guarded; long grace because a legitimate WBPP
    # run can exceed an hour — but a live one keeps its newest mtime fresh
    # and its files open.
    DebrisSpec(
        name="wbpp_staging",
        dirs=_temp_dirs,
        pattern="tmp*",
        min_age_s=6 * 3600,
        guard=_looks_like_wbpp_staging,
    ),
    # Small named staging dirs: the pix runner's SCP sentinel and the
    # rc-astro output staging (rcastro_bxt_ / rcastro_sxt_ / rcastro_nxt_).
    DebrisSpec(
        name="pix_sentinel",
        dirs=_temp_dirs,
        pattern="pix-wbpp-*",
        min_age_s=6 * 3600,
    ),
    DebrisSpec(
        name="rcastro_staging",
        dirs=_temp_dirs,
        pattern="rcastro_*",
        min_age_s=6 * 3600,
    ),
)


def _size_of(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file()
            )
        return path.stat().st_size
    except OSError:
        return 0


def reap(specs: tuple[DebrisSpec, ...] = SPECS, now: float | None = None) -> dict:
    """One janitor tick. Returns {spec_name: {"deleted": n, "bytes": b}}."""
    now = time.time() if now is None else now
    report: dict[str, dict[str, int]] = {}
    for spec in specs:
        try:
            if not spec.enabled():
                continue
            candidates = [
                p
                for d in spec.dirs()
                for p in d.glob(spec.pattern)
                if now - _newest_mtime(p) >= spec.min_age_s and spec.guard(p)
            ]
            live = _open_paths(candidates)
            deleted = 0
            freed = 0
            for path in candidates:
                if path in live:
                    continue
                size = _size_of(path)
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                except OSError as e:
                    log.warning("janitor delete failed", path=str(path), error=str(e))
                    continue
                deleted += 1
                freed += size
            if deleted:
                report[spec.name] = {"deleted": deleted, "bytes": freed}
                log.info(
                    "janitor reaped debris",
                    spec=spec.name,
                    deleted=deleted,
                    gb=round(freed / 1e9, 2),
                )
        except Exception:
            # A broken spec must never take down the tick (or the loop).
            import traceback
            log.warning(
                "janitor spec failed", spec=spec.name, error=traceback.format_exc()
            )
    return report


def run_forever(interval_s: float = DEFAULT_INTERVAL_S) -> None:
    """Blocking janitor loop; run in a daemon thread. First tick immediate
    (complements the make-start shell sweep, which covers pre-launch)."""
    while True:
        reap()
        time.sleep(interval_s)


if __name__ == "__main__":
    print(reap())
