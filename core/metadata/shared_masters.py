"""Shared master folders: other people's masters, read like a raw folder.

A shared master folder (`context.shared_dirs`; a Dropbox folder, a synced
bucket — AstroFlow never learns which) holds masters published by every
AstroFlow instance in a group (`stacks.publish_shared`), and whatever else a
member drops in: a master from another program is foreign all the same. This
module READS such a folder and derives what it must: a master that brings no
sidecar is plate-solved and measured in place, the sidecar written beside it
exactly as in a raw data folder, so the first member to see it does the work
for all through the sync tool. The master itself is never modified.

Two things differ from a raw folder, because a shared folder has many
writers and a sync tool between them. A file can change under us — it
arrives in pieces, or is re-pushed under its name — so a sidecar is trusted
only for the bytes it recorded: its size must be the file's, and when THIS
machine's scan sees the file's size or mtime move, the pixels are hashed
once and compared with the sidecar's content hash before the sidecar is
believed (a sync tool's mtimes are its own business, so no machine's mtime
is ever written into the shared sidecar, and only a genuine change costs a
read). A file nobody has described yet is measured only once it has sat
unchanged since the previous scan or is minutes old, never while arriving.
And two members may write the same sidecar at once; the sync tool keeps one
as a conflicted copy, which pairs with no data file and is ignored. The
pieces and their union follow the raw library (`metadata.image_metadata`):
per-folder, fingerprint keyed, disconnected = unchanged.
"""
from __future__ import annotations

from pathlib import Path

import structlog

from cache.context_ref import ContextRef
from decorators import node
from hashing import hash_call
from metadata.library_scan import ScanRecord

log = structlog.get_logger()

_IMAGE_SUFFIXES = (".xisf", ".fits", ".fit", ".tiff", ".tif")


def _sidecar_data_path(sidecar: Path) -> Path | None:
    """`foo.xisf.json` → `foo.xisf`, when that is an image file name."""
    if sidecar.suffix != ".json":
        return None
    data = sidecar.with_suffix("")
    return data if data.suffix.lower() in _IMAGE_SUFFIXES else None


def scan_shared_dir(dir: str | Path) -> ScanRecord:
    """Fresh scan of a PRESENT shared folder: every image file, recursively,
    skipping dot-prefixed names (a sync tool's temporaries and conflict
    markers). The listing carries the DATA files' size and mtime, so the
    scan delta speaks of the masters alone; the sidecars' stats enter the
    fingerprint, so a sidecar someone else filled in re-keys the piece."""
    root = Path(dir)
    entries: list[tuple[str, int, int]] = []
    sidecars: list[tuple[str, int, int]] = []
    for data in root.rglob("*"):
        if data.suffix.lower() not in _IMAGE_SUFFIXES or not data.is_file():
            continue
        if any(part.startswith(".") for part in data.relative_to(root).parts):
            continue
        try:
            st = data.stat()
            rel = str(data.relative_to(root))
            entries.append((rel, st.st_size, st.st_mtime_ns))
            sidecar = data.with_suffix(data.suffix + ".json")
            if sidecar.exists():
                ss = sidecar.stat()
                sidecars.append((rel, ss.st_size, ss.st_mtime_ns))
        except OSError:
            continue  # vanished mid-scan → the next scan sees the folder as it is
    listing = tuple(sorted(entries))
    return ScanRecord(dir=str(root), listing=listing,
                      fingerprint=hash_call("shared_listing_v2", listing, tuple(sorted(sidecars))))


#: A file nobody has described is measured once it is this old, even on
#: first sight: the sync tool has long finished with it.
SETTLED_AFTER_S = 10 * 60


def sidecar_is_current(data: Path, size: int) -> bool:
    """A shared file's sidecar exists and recorded this size."""
    from file_io.sidecar import get_file_size, sidecar_path_for
    return sidecar_path_for(data).exists() and get_file_size(data) == size


def measurements_due(root: Path, old, record, now: float | None = None) -> list[tuple]:
    """`(path, size, mtime_ns, verify)` for the listed files this scan must
    look at: a file without a current sidecar, once settled (unchanged since
    the previous scan, or `SETTLED_AFTER_S` old); and a file whose size or
    mtime moved since the previous scan although its sidecar is current —
    verified by hash before the sidecar is believed."""
    import time
    from metadata.library_scan import delta
    now = time.time() if now is None else now
    added, changed, _removed = delta(old, record)
    unsettled = set(added) | set(changed)
    due = []
    for rel, size, mtime in record.listing:
        path = root / rel
        if sidecar_is_current(path, size):
            if rel in changed:
                due.append((str(path), size, mtime, True))
        elif rel not in unsettled or now - mtime / 1e9 > SETTLED_AFTER_S:
            due.append((str(path), size, mtime, False))
    return due


@node(num_cpus=0.25, auto_memory=True)
def compute_shared_master(path: str, size: int, mtime_ns: int, verify: bool) -> None:
    """Measure a shared master in place — content hash, header, statistics,
    plate solution — writing its sidecar beside it, as `compute_image_metadata`
    does for a raw frame. Keyed on the file's size and mtime as this machine
    saw them. With `verify`, the sidecar is current by size but the file moved
    under it: the pixels are hashed and the sidecar kept if it describes them,
    dropped and the file measured anew if not (an AstroFlow-published master
    is its name, so it is never re-measured for a stat change). A dropped
    sidecar takes any broken marker with it: a truncated file that has since
    finished arriving must be retried."""
    from file_io.generic import hash_of_image_data
    from file_io.sidecar import sidecar_path_for
    from metadata.image_metadata import compute_image_metadata
    from metadata.sidecar_io import read_image_metadata
    data = Path(path)
    if sidecar_is_current(data, size):
        if not verify:
            return
        try:
            known = read_image_metadata(data)
        except Exception:
            known = None
        if known is not None and (known.header.get("CALLHASH") or hash_of_image_data(data) == known.content_hash):
            return
        log.info("shared master changed under its sidecar — measuring again", file=path)
    sidecar_path_for(data).unlink(missing_ok=True)
    from file_io.generic import read_header
    from plate_solving import PlateSolveFailedException
    try:
        compute_image_metadata._function(data, plate_solve=declares_light(read_header(data)))
    except PlateSolveFailedException:
        # Recorded unsolved, as a raw frame would be: described, never admitted.
        compute_image_metadata._function(data, plate_solve=False)


def declares_light(header: dict) -> bool:
    """A shared folder holds light masters: a file says otherwise only with
    an IMAGETYP that is not a light (every calibration master carries one;
    survey products often carry none)."""
    from metadata.image_types import is_light_frame
    return "IMAGETYP" not in (header or {}) or is_light_frame(header)


def _is_shared_master(meta, size_on_disk: int) -> bool:
    """A file this folder may lend to a mosaic: a solved, single-band light
    master whose bytes are all there (the sidecar recorded this size)."""
    from file_io.sidecar import get_file_size
    return bool(
        declares_light(meta.header) and meta.plate_solution is not None
        and meta.channels == 1 and get_file_size(meta.filename) == size_on_disk
    )


def admissible_masters(root: Path, listing: tuple) -> dict:
    """`{content_hash: ImageMetadata}` of the listed files that are whole,
    solved, single-band light masters, from their sidecars."""
    from metadata.sidecar_io import read_image_metadata

    masters: dict = {}
    for rel, size, _mtime in listing:
        f = Path(root) / rel
        try:
            meta = read_image_metadata(f)
        except Exception as e:  # a sidecar someone else is still writing
            log.warning("shared sidecar unreadable", file=str(f), error=str(e))
            continue
        if meta is None or not isinstance(meta.content_hash, str):
            continue
        if not _is_shared_master(meta, size):
            continue
        masters.setdefault(meta.content_hash, meta)   # duplicate bytes — first wins
    return masters


@node(num_cpus=0.25, resources={"diskcache": 0.001})
def build_shared_library(dir: str, fingerprint: str, listing: tuple) -> ContextRef:
    """One shared folder's piece: `{content_hash: ImageMetadata}` of its
    admissible masters, from their sidecars alone. Cached on (dir,
    fingerprint, listing), exactly like `build_dir_library`."""
    masters = admissible_masters(Path(dir), listing)
    log.info("shared masters piece", dir=dir, listed=len(listing), admissible=len(masters))
    identity = tuple(sorted((h, str(m.filename)) for h, m in masters.items()))
    return ContextRef.store_large(masters, identity)


@node(num_cpus=0.25)
def build_shared_dicts(*pieces: ContextRef) -> ContextRef:
    """The union of the shared folders' pieces, in canonical order."""
    raw: dict = {}
    for piece in pieces:
        for content_hash, meta in piece.get().items():
            raw.setdefault(content_hash, meta)
    merged = {h: raw[h] for h in sorted(raw)}
    identity = tuple(sorted((h, str(m.filename)) for h, m in merged.items()))
    return ContextRef.store_large(merged, identity)
