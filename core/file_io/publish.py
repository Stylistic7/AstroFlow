"""Atomic publication of a produced file to its final content-addressed path.

The rule this module exists to enforce: **a file at its final path is
complete**. Everything downstream leans on that — the cache treats an existing
output as satisfied, the tile server treats it as viewable, and the sweep
treats it as live.

A bare `shutil.copy`/`move` breaks the rule whenever the source is on a
DIFFERENT filesystem (a `TemporaryDirectory`, PixInsight's working dir), because
the copy then writes incrementally straight into the final path. A reader can
observe a half-written file, and under re-dispatch two publishers race into the
same `HASH.ext` and one truncates the other — a file cut off at the
image/metadata boundary that never satisfies the cache, so it re-derives
forever (the "BXT loop", and the same shape as the 2026-08-14 pyramid
incident).

Copy to a unique staging name ON THE DESTINATION'S FILESYSTEM, then
`os.replace`: a same-filesystem replace is atomic, so a reader never sees a
partial file and two publishers both land complete files (last wins) rather
than one torn one.

Extracted from `rc_astro_cli._move`, which was the only producer doing this
correctly.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def publish_atomically(produced: Path, dest: Path, *, consume: bool = False) -> None:
    """Publish `produced` to `dest` so `dest` is never observed incomplete.

    `consume=True` deletes the source afterwards (what `shutil.move` did).
    Safe when `produced` and `dest` are the same path — a no-op — so callers
    that sometimes produce in place don't need to special-case it.
    """
    produced, dest = Path(produced), Path(dest)
    if not produced.exists():
        raise FileNotFoundError(f"nothing to publish at {produced}")
    if produced.resolve() == dest.resolve():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.with_name(f"{dest.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        shutil.copy2(str(produced), str(staging))  # full copy onto dest's filesystem
        os.replace(str(staging), str(dest))        # atomic same-fs publish
    finally:
        staging.unlink(missing_ok=True)            # cleanup if copy/replace failed
    if consume:
        produced.unlink(missing_ok=True)


def publish_pair_atomically(
    primary_src: Path, primary_dest: Path,
    sidecar_src: Path | None, sidecar_dest: Path | None,
) -> None:
    """Publish a payload and its sidecar, **sidecar first, payload last**.

    Two `os.replace` calls cannot be made one atom (POSIX has no multi-rename
    transaction; `renameat2`/`renamex_np` only SWAP two existing paths). So
    order them: whichever file gates visibility lands last. The payload is the
    gate, so a crash between the renames leaves an orphan sidecar — harmless —
    rather than a payload whose sidecar is missing.
    """
    if sidecar_src is not None and sidecar_dest is not None and Path(sidecar_src).exists():
        publish_atomically(Path(sidecar_src), Path(sidecar_dest))
    publish_atomically(primary_src, primary_dest)
