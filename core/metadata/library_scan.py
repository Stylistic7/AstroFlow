"""Per-input-dir library scan records: the change-detection primitive that
makes startup instant when the user's library hasn't changed.

One fresh `find_image_files` walk per input dir yields a LISTING —
`((relpath, size, mtime_ns), ...)` sorted — and its FINGERPRINT (a stable
hash). The fingerprint keys every downstream pure computation (per-dir
metadata/defect pieces, the delta of `compute_image_metadata` nodes), so an
unchanged dir costs one directory walk and a handful of diskcache hits — no
per-file Ray dispatches, no sidecar sweeps.

DISCONNECTED SEMANTICS (a design requirement, not an edge case): a dir whose
volume is unmounted means "this disk hasn't changed". `scan_or_reuse` returns
the last persisted record verbatim — same fingerprint, same listing — so
every fingerprint-keyed computation cache-hits, frames on the absent disk
remain first-class members of the metadata, stack identities don't move, and
the UI stays fully functional off `Processed`. Only a node that must actually
READ a file from the missing disk fails, locally. The durable record is
justified against the avoid-derived-state rule because there is no
authoritative source to consult while the media is offline: present → the
scan wins; absent → the record is by definition the best knowledge. Drift is
structurally impossible.

Known blind spot (accepted): an in-place edit that preserves size AND mtime
is invisible — the same trust rsync places in stat(). Astro libraries are
append-only capture drops, and this is strictly better than the status quo
(`compute_image_metadata` keys on the filename alone, so today ANY in-place
change cache-hits stale forever; the fingerprint delta adds size/mtime change
detection).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from file_io.find_image_files import find_image_files
from hashing import hash_call

log = structlog.get_logger()


def _context():
    # Lazy: importing `context` at module load drags Ray into every
    # importer, and the pure pieces of this module (scan_dir, delta) must
    # stay unit-testable without a cluster.
    from context import context
    return context


@dataclass(frozen=True)
class ScanRecord:
    """One dir's library state: `listing` is sorted
    ((relpath, size, mtime_ns), ...); `fingerprint` is its stable hash."""
    dir: str
    fingerprint: str
    listing: tuple[tuple[str, int, int], ...]

    def paths(self) -> list[Path]:
        root = Path(self.dir)
        return [root / rel for rel, _, _ in self.listing]


def scan_dir(dir: str | Path) -> ScanRecord:
    """Fresh scan of a PRESENT dir. Uses the same `find_image_files`
    filter as the old scatter glob (masters included), so the file set is
    identical to what the pipeline would previously have dispatched."""
    root = Path(dir)
    entries: list[tuple[str, int, int]] = []
    for f in find_image_files(str(root), include_masters=True):
        p = Path(f)
        try:
            st = p.stat()
        except OSError:
            continue  # vanished mid-scan → next run's delta picks it up
        entries.append((str(p.relative_to(root)), st.st_size, st.st_mtime_ns))
    listing = tuple(sorted(entries))
    return ScanRecord(
        dir=str(root),
        # The same canonical call-hasher every @node uses — deterministic
        # across processes/runs for plain tuples.
        fingerprint=hash_call("library_listing_v1", listing),
        listing=listing,
    )


def _record_key(dir: str | Path) -> tuple:
    return ("library_scan", str(dir))


def load_record(dir: str | Path) -> ScanRecord | None:
    raw = _context().get(_record_key(dir))
    if not isinstance(raw, dict):
        return None
    try:
        return ScanRecord(
            dir=str(dir),
            fingerprint=raw["fingerprint"],
            listing=tuple(tuple(e) for e in raw["listing"]),
        )
    except (KeyError, TypeError):
        return None


def store_record(record: ScanRecord) -> None:
    _context()[_record_key(record.dir)] = {
        "fingerprint": record.fingerprint,
        "listing": [list(e) for e in record.listing],
    }


def dir_available(dir: str | Path) -> bool:
    """False ONLY for the disconnected-volume case. A dir under /Volumes
    whose volume root is unmounted is "absent" (reuse the record); a
    present volume with the dir deleted/emptied is a REAL change."""
    root = Path(dir)
    if root.is_dir():
        return True
    parts = root.parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "Volumes":
        volume = Path("/Volumes") / parts[2]
        return volume.exists()  # volume mounted, dir truly gone → "present"
    if root.drive:
        # Windows: an unplugged disk's letter, or an offline share's \\host\share
        return Path(root.anchor).exists()
    return True  # non-volume path missing = real (empty) state, not absence


def scan_or_reuse(dir: str | Path, scanner=None) -> tuple[ScanRecord, bool]:
    """The startup entry point: (record, fresh).

    Present dir → fresh scan, return (record, True). Disconnected volume →
    last record verbatim, (record, False) — semantically "this disk hasn't
    changed". First-ever sight of a disconnected dir → empty record
    (nothing known), warn once.

    Deliberately does NOT persist: the record must be stored only AFTER the
    delta's compute nodes and the per-dir pieces succeed (the publisher
    terminal does it). Persisting at scan time would swallow the delta
    forever if the run died in between; deferring makes a dead run
    self-healing — the next start re-deltas the same files, whose op-level
    cache entries make the re-dispatch cheap.
    """
    if dir_available(dir):
        return (scanner or scan_dir)(dir), True
    known = load_record(dir)
    if known is not None:
        log.info(
            "library dir unavailable — treating as unchanged",
            dir=str(dir), files=len(known.listing),
        )
        return known, False
    log.warning(
        "library dir unavailable and never scanned — treating as empty",
        dir=str(dir),
    )
    return ScanRecord(dir=str(dir), fingerprint="never-scanned", listing=()), False


def delta(
    old: ScanRecord | None, new: ScanRecord,
) -> tuple[list[str], list[str], list[str]]:
    """(added, changed, removed) relpaths between two records of one dir.

    `changed` = same relpath, different (size, mtime_ns). INFORMATIONAL
    ONLY since 2026-08-13: the scatter no longer acts on it. Acting on it
    meant deleting the frame's sidecar and evicting its cache entry, and an
    interrupted pass then left the frame with neither — invisible to the
    sidecar-sourced library. Filesystem metadata is not trustworthy enough
    to justify destroying the only copy of a solve: a network mount, a copy
    tool, or a clock can all perturb size/mtime without a pixel changing.
    Re-solving after a genuine edit is an explicit user action."""
    old_by_path = {rel: (sz, mt) for rel, sz, mt in (old.listing if old else ())}
    new_by_path = {rel: (sz, mt) for rel, sz, mt in new.listing}
    added = [rel for rel in new_by_path if rel not in old_by_path]
    changed = [
        rel for rel, stat in new_by_path.items()
        if rel in old_by_path and old_by_path[rel] != stat
    ]
    removed = [rel for rel in old_by_path if rel not in new_by_path]
    return added, changed, removed
