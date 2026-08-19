"""Directory-affinity calibration matching.

When a light stack's frames live under a directory tree that contains
externally pre-stacked masters (typically `Calibration Masters/Master*.fit`),
those masters are the user's chosen calibration for that target. Find them
by walking up from each frame's parent directory.

Used as the *first-tried default* by the CALIBRATION_BIAS / CALIBRATION_DARK /
CALIBRATION_FLAT decision points in `decisions/resolve.py`; the chronological
matcher is the fallback when no directory-affinity match exists. The user can
override either via the standard decision-override API.

External masters are deliberately *not* picked up by raw-frame discovery
(`find_image_files` excludes `master*` files), so we re-scan the filesystem
here on demand. Per-directory results are cached for the life of the process.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import astropy.io.fits as pyfits

from metadata.image_types import (
    is_bias_frame, is_dark_frame, is_flat_frame, is_master,
)


SensorShape = tuple[int | None, int | None]   # (NAXIS1, NAXIS2)


@dataclass(frozen=True)
class ExternalMaster:
    path: str
    image_type: str            # 'bias' | 'dark' | 'flat'
    shape: SensorShape         # the pixel array, not a claim about it
    filter: str | None
    exposure: float | None
    temperature: float | None


# Directory names that say "the calibration for the frames next door lives
# here". Used only to ORDER candidate directories, never to gate them, so a
# false positive costs nothing and an unnamed folder still gets scanned.
_CALIBRATION_DIR_HINTS = (
    "calibration", "calibrate", "calib", "cal",
    "master", "bias", "dark", "flat",
)


def _looks_like_calibration_dir(name: str) -> bool:
    return any(hint in name.lower() for hint in _CALIBRATION_DIR_HINTS)


def _classify_image_type(header: dict) -> str | None:
    if not is_master(header):
        return None
    if is_bias_frame(header):
        return "bias"
    if is_dark_frame(header):
        return "dark"
    if is_flat_frame(header):
        return "flat"
    return None


def _read_master_metadata(path: Path) -> ExternalMaster | None:
    try:
        with pyfits.open(str(path), memmap=False) as hdul:
            header = dict(hdul[0].header)
    except Exception:
        return None
    image_type = _classify_image_type(header)
    if image_type is None:
        return None
    temp = header.get("CCD-TEMP", header.get("SET-TEMP"))
    return ExternalMaster(
        path=str(path.absolute()),
        image_type=image_type,
        shape=(header.get("NAXIS1"), header.get("NAXIS2")),
        filter=header.get("FILTER"),
        exposure=float(header["EXPOSURE"]) if "EXPOSURE" in header else None,
        temperature=float(temp) if temp is not None else None,
    )


_dir_master_cache: dict[str, list[ExternalMaster]] = {}


def masters_in_dir(directory: Path) -> list[ExternalMaster]:
    """All `master*.fit[s]` files directly in `directory` (non-recursive).

    Memoized for the life of the process. Cache invalidation is handled by
    the broader compute graph re-entering on `drain_and_restart`, which
    rebuilds the worker process and clears the cache with it.

    Returns an empty list (and caches it) for directories we can't read or
    which don't exist — the walk happily passes those by.
    """
    key = str(directory.resolve())
    if key in _dir_master_cache:
        return _dir_master_cache[key]
    out: list[ExternalMaster] = []
    try:
        if directory.is_dir():
            for entry in sorted(directory.iterdir()):
                if entry.suffix.lower() not in (".fit", ".fits"):
                    continue
                if not entry.name.lower().startswith("master"):
                    continue
                m = _read_master_metadata(entry)
                if m is not None:
                    out.append(m)
    except (PermissionError, OSError):
        pass
    _dir_master_cache[key] = out
    return out


def _ancestor_dirs(start: Path) -> Iterable[Path]:
    """Yield `start` and each ancestor up to the filesystem root."""
    p = start.resolve()
    yielded: set[Path] = set()
    while p not in yielded:
        yielded.add(p)
        yield p
        if p.parent == p:
            break
        p = p.parent


def _match_kind(
    masters: list[ExternalMaster],
    kind: str,
    shape: SensorShape | None,
    temperature: float | None,
    filter: str | None,
    exposure: float | None,
) -> list[ExternalMaster]:
    """Filter masters in a directory down to those that match the run's needs.

    Mirrors the constraints used by the chronological matcher in
    `decisions/resolve.py`: same sensor geometry, similar temperature, dark
    exposure must be ≥ light exposure, flat must match filter.

    Geometry replaced an `INSTRUME` string equality test here, and that test
    was a live correctness bug: when a driver reconnect renamed one camera
    mid-session, four Starbase light stacks failed to match the flats sitting
    in their own `Calibration Masters/` folder and were integrated
    uncalibrated. A master that does not share the light's pixel dimensions
    genuinely cannot calibrate it; a master that merely disagrees about its
    own name calibrates it fine.
    """
    cands = [
        m for m in masters
        if m.image_type == kind
        and (shape is None or None in m.shape or m.shape == shape)
    ]
    if kind in ("bias", "dark"):
        cands = [
            m for m in cands
            if (m.temperature is None or temperature is None
                or abs(m.temperature - temperature) < 10.0)
        ]
    if kind == "dark":
        cands = [
            m for m in cands
            if (m.exposure is None or exposure is None
                or m.exposure + 1.0 >= exposure)
        ]
    if kind == "flat":
        cands = [m for m in cands if filter is None or m.filter == filter]
    return cands


def _candidate_dirs_at(ancestor: Path) -> Iterable[Path]:
    """The ancestor itself, plus its immediate subdirectories — calibration-
    named subdirectories first.

    Catches the canonical layout where lights live in `<target>/Lights/` and
    masters live in a sibling `<target>/Calibration Masters/`. Scanning one
    level deep at each ancestor is enough — anything deeper than that is
    almost certainly not the user's intended association.

    A folder called `Calibration Masters/` sitting beside the lights is the
    clearest statement of intent the filesystem can carry, so it is searched
    before an unnamed sibling that merely happens to hold master files.
    Ordering only: every subdirectory is still searched, and `sorted` keeps
    the walk deterministic within each group.
    """
    yield ancestor
    try:
        if ancestor.is_dir():
            children = sorted(
                c for c in ancestor.iterdir()
                if c.is_dir() and not c.name.startswith(".")
            )
            yield from (c for c in children if _looks_like_calibration_dir(c.name))
            yield from (c for c in children if not _looks_like_calibration_dir(c.name))
    except (PermissionError, OSError):
        pass


def find_external_master(
    run_frame_filenames: Iterable[str],
    *,
    kind: str,                       # 'bias' | 'dark' | 'flat'
    shape: SensorShape | None,
    temperature: float | None,
    filter: str | None,
    exposure: float | None,
) -> str | None:
    """Walk up from each frame's parent directory looking for a master that
    matches the run's calibration needs. At each ancestor, also peek one level
    down into its immediate subdirectories (catches the canonical
    `<target>/Lights/` + `<target>/Calibration Masters/` layout). Returns
    the path of the first match (nearest ancestor wins), or None.
    """
    starting_dirs = {Path(f).resolve().parent for f in run_frame_filenames}
    if not starting_dirs:
        return None

    visited: set[Path] = set()
    for start in starting_dirs:
        for ancestor in _ancestor_dirs(start):
            for candidate_dir in _candidate_dirs_at(ancestor):
                if candidate_dir in visited:
                    continue
                visited.add(candidate_dir)
                cands = _match_kind(
                    masters_in_dir(candidate_dir),
                    kind, shape, temperature, filter, exposure,
                )
                if cands:
                    return cands[0].path
    return None
