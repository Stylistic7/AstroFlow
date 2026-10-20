"""Shared master folders: other people's masters, read from their sidecars.

A shared master folder (`context.shared_dirs`; a Dropbox folder, a synced
bucket — AstroFlow never learns which) holds post-stack-calibrated masters
and their sidecars, published by every AstroFlow instance in a group
(`stacks.publish_shared`). This module READS such a folder; nothing here
writes into one, and nothing in the app writes a sidecar beside a file it
found there: the producer's sidecar is the manifest, authoritative and
complete (metadata, plate solution, calibration certificate, pipeline stamp),
so no master is ever opened, solved or measured on the consumer's side. A
consumer that wrote beside a shared file would race every other consumer
through the sync tool (conflicted copies, sync storms).

The scan lists the DATA files that have a sidecar, with the pair's size and
newest mtime, so a master still arriving through the sync tool (size short of
what its sidecar recorded) is left out until it is whole, and a change to
either file re-keys the folder's piece. The pieces and their union follow the
raw library exactly (`metadata.image_metadata`): per-folder, fingerprint
keyed, disconnected = unchanged.
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
    """Fresh scan of a PRESENT shared folder: every image file beside a
    sidecar, recursively, skipping dot-prefixed names (a sync tool's
    temporaries and conflict markers)."""
    root = Path(dir)
    entries: list[tuple[str, int, int]] = []
    for sidecar in root.rglob("*.json"):
        if any(part.startswith(".") for part in sidecar.relative_to(root).parts):
            continue
        data = _sidecar_data_path(sidecar)
        if data is None:
            continue
        try:
            ds, ss = data.stat(), sidecar.stat()
        except OSError:
            continue  # vanished mid-scan → the next scan sees the folder as it is
        entries.append((str(data.relative_to(root)), ds.st_size,
                        max(ds.st_mtime_ns, ss.st_mtime_ns)))
    listing = tuple(sorted(entries))
    return ScanRecord(dir=str(root), fingerprint=hash_call("shared_listing_v1", listing),
                      listing=listing)


def _is_shared_master(meta, size_on_disk: int) -> bool:
    """A file this folder may lend to a mosaic: a stamped, solved light master
    (`PIPEHASH`, `CALLHASH`, `STBLHASH`) whose bytes are all there."""
    from file_io.sidecar import get_file_size
    from hashing.pipeline_stamp import stamp_of
    from metadata.image_types import is_light_frame
    h = meta.header
    return bool(
        stamp_of(h) and h.get("CALLHASH") and h.get("STBLHASH")
        and is_light_frame(h) and meta.plate_solution is not None
        and get_file_size(meta.filename) == size_on_disk
    )


def admissible_masters(root: Path, listing: tuple) -> dict:
    """`{content_hash: ImageMetadata}` of the listed files that are whole,
    stamped, solved light masters, from their sidecars alone."""
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
