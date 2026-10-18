"""Re-save the cache entries Python 3.13 pickled with `pathlib._local`, so
Python 3.12 can read them (2026-10-17, the move to 3.12 everywhere).

Python 3.13 moved pathlib's classes into `pathlib._local`, and every Path it
pickled names that module, which 3.12 does not have. This finds every stored
pickle, key or value, inline or in a file, that contains the bytes
`pathlib._local`, loads it with `pathlib._local` aliased to `pathlib`, and
stores it again: the new pickle names plain `pathlib`, which 3.12 and 3.13 both
read. Every other entry is left byte-for-byte as it is.

Run with the app STOPPED, under Python 3.12, after backing the store up:

    python scripts/migrate_pathlib_local_pickles.py <diskcache dir> [--apply]

Without --apply it only counts. `--verify` loads every entry with no alias.
"""
from __future__ import annotations

import pathlib
import pickle
import sqlite3
import sys
import types
from pathlib import Path

import diskcache

NEEDLE = b"pathlib._local"


def _alias() -> None:
    """`pathlib._local` as 3.13 named it, resolved to 3.12's pathlib."""
    if "pathlib._local" in sys.modules:
        return
    local = types.ModuleType("pathlib._local")
    for name in ("PurePath", "PurePosixPath", "PureWindowsPath", "Path", "PosixPath", "WindowsPath"):
        setattr(local, name, getattr(pathlib, name))
    sys.modules["pathlib._local"] = local


def _value_bytes(directory: Path, row) -> bytes | None:
    """The stored value's raw bytes (inline blob or its file); None when the
    value is not a pickle (diskcache stores raw bytes and text as-is)."""
    _rowid, _key, _raw, mode, filename, value = row
    if mode != diskcache.core.MODE_PICKLE:
        return None
    if filename is not None:
        return (directory / filename).read_bytes()
    return bytes(value) if value is not None else None


def main(directory: str, apply: bool, verify: bool) -> int:
    d = Path(directory)
    db = sqlite3.connect(str(d / "cache.db"))
    rows = db.execute("SELECT rowid, key, raw, mode, filename, value FROM Cache").fetchall()
    db.close()
    affected = []
    for row in rows:
        rowid, key, raw = row[0], row[1], row[2]
        key_hit = not raw and isinstance(key, (bytes, memoryview)) and NEEDLE in bytes(key)
        value = _value_bytes(d, row)
        if key_hit or (value is not None and NEEDLE in value):
            affected.append((rowid, key_hit))
    print(f"{d}: {len(rows)} entries, {len(affected)} name pathlib._local "
          f"({sum(k for _, k in affected)} in the key)")
    if verify:
        return _verify(d)
    if not apply or not affected:
        return 0

    _alias()
    cache = diskcache.Cache(str(d), eviction_policy="none")
    db = sqlite3.connect(str(d / "cache.db"))
    done, dead = 0, []
    for rowid, key_hit in affected:
        key_blob, raw = db.execute("SELECT key, raw FROM Cache WHERE rowid = ?", (rowid,)).fetchone()
        try:
            key = key_blob if raw else pickle.loads(bytes(key_blob))
            value = cache.get(key, default=None, retry=True)
        except Exception as e:
            # Unloadable for another reason too (a class whose module was
            # deleted): already dead under 3.13. Left exactly as it is.
            dead.append((rowid, f"{type(e).__name__}: {str(e)[:100]}"))
            continue
        if key_hit:
            # The key itself re-pickles differently: drop the old row first.
            db.execute("DELETE FROM Cache WHERE rowid = ?", (rowid,))
            db.commit()
        cache.set(key, value, retry=True)
        done += 1
    db.close()
    cache.close()
    print(f"re-saved {done} entries; left {len(dead)} that no Python can load (deleted code)")
    for rowid, why in dead[:10]:
        print(f"  rowid {rowid}: {why}")
    return _verify(d)


def _verify(d: Path) -> int:
    """Every pickle loads without the alias (run in a fresh 3.12 process)."""
    if "pathlib._local" in sys.modules:
        print("verify must run in a fresh process (the alias is installed here)")
        return 2
    import collections
    db = sqlite3.connect(str(d / "cache.db"))
    causes = collections.Counter()
    for row in db.execute("SELECT rowid, key, raw, mode, filename, value FROM Cache"):
        try:
            if not row[2]:
                pickle.loads(bytes(row[1]))
            value = _value_bytes(d, row)
            if value is not None:
                pickle.loads(value)
        except Exception as e:
            causes[f"{type(e).__name__}: {str(e)[:90]}"] += 1
    db.close()
    local = sum(n for c, n in causes.items() if "pathlib._local" in c)
    print(f"verify: {sum(causes.values())} entries fail to load, {local} of them for pathlib._local")
    for cause, n in causes.most_common(8):
        print(f"  {n:6d}  {cause}")
    return 1 if local else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0], "--apply" in sys.argv, "--verify" in sys.argv))
