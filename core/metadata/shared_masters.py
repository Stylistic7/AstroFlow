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
arrives in pieces, or is re-pushed under its name — so a sidecar here is
valid only while the size it recorded is the file's size; otherwise it is
discarded and the file measured again (`compute_shared_master` keys on size
and mtime). And two members may write the same sidecar at once; the sync
tool keeps one as a conflicted copy, which pairs with no data file and is
ignored. The pieces and their union follow the raw library
(`metadata.image_metadata`): per-folder, fingerprint keyed, disconnected =
unchanged.
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
    with the newest mtime of the file and its sidecar, skipping dot-prefixed
    names (a sync tool's temporaries and conflict markers)."""
    root = Path(dir)
    entries: list[tuple[str, int, int]] = []
    for data in root.rglob("*"):
        if data.suffix.lower() not in _IMAGE_SUFFIXES or not data.is_file():
            continue
        if any(part.startswith(".") for part in data.relative_to(root).parts):
            continue
        try:
            st = data.stat()
            mtime = st.st_mtime_ns
            sidecar = data.with_suffix(data.suffix + ".json")
            if sidecar.exists():
                mtime = max(mtime, sidecar.stat().st_mtime_ns)
        except OSError:
            continue  # vanished mid-scan → the next scan sees the folder as it is
        entries.append((str(data.relative_to(root)), st.st_size, mtime))
    listing = tuple(sorted(entries))
    return ScanRecord(dir=str(root), fingerprint=hash_call("shared_listing_v1", listing),
                      listing=listing)


MEASURED = "measured"   # sidecar section: the bytes a shared file's sidecar describes


def sidecar_is_current(data: Path, size: int) -> bool:
    """A shared file's sidecar describes THESE bytes. A file we measured here
    recorded its size AND mtime (`MEASURED`): a re-pushed master, even one of
    exactly the same size, has a new mtime and is measured again. A sidecar
    that arrived with an AstroFlow-published master recorded the size alone,
    and that master's name is its content: new bytes come under a new name.
    A half-arrived file fails either test."""
    from file_io.sidecar import get_file_size, get_section, sidecar_path_for
    if not sidecar_path_for(data).exists():
        return False
    measured = get_section(data, MEASURED)
    if isinstance(measured, dict):
        try:
            st = data.stat()
        except OSError:
            return False
        return measured.get("size") == st.st_size and measured.get("mtime_ns") == st.st_mtime_ns
    return get_file_size(data) == size


@node(num_cpus=0.25, auto_memory=True)
def compute_shared_master(path: str, size: int, mtime_ns: int) -> None:
    """Measure a shared master in place — content hash, header, statistics,
    plate solution — writing its sidecar beside it, as `compute_image_metadata`
    does for a raw frame, then record which bytes it describes. Keyed on the
    file's size and mtime, so a file that changed is measured anew: a sidecar
    that recorded other bytes is dropped first, and with it any broken marker
    (a truncated file that has since finished arriving must be retried)."""
    from file_io.sidecar import set_section, sidecar_path_for
    from metadata.image_metadata import compute_image_metadata
    data = Path(path)
    if sidecar_is_current(data, size):
        return
    sidecar_path_for(data).unlink(missing_ok=True)
    st = data.stat()
    compute_image_metadata._function(data)
    set_section(data, MEASURED, {"size": st.st_size, "mtime_ns": st.st_mtime_ns})


def _is_shared_master(meta, size_on_disk: int) -> bool:
    """A file this folder may lend to a mosaic: a solved, single-band light
    master whose bytes are all there (the sidecar recorded this size)."""
    from file_io.sidecar import get_file_size
    from metadata.image_types import is_light_frame
    return bool(
        is_light_frame(meta.header) and meta.plate_solution is not None
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
