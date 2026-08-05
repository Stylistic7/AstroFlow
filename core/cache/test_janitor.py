"""The wall-clock janitor: deletes only provably-abandoned debris.

Every test builds its own DebrisSpec against tmp_path — no app imports, no
Ray, no real Darwin dirs — except the lsof test, which exercises the real
`_open_paths` against a genuinely held-open file (macOS always has lsof).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from cache.janitor import (
    DebrisSpec,
    _looks_like_wbpp_staging,
    _open_paths,
    reap,
)

OLD = 10_000.0  # seconds — comfortably past every grace period used here


def _spec(tmp_path: Path, **overrides) -> DebrisSpec:
    defaults = dict(
        name="test_debris",
        dirs=lambda: [tmp_path],
        pattern="*.junk",
        min_age_s=60.0,
    )
    return DebrisSpec(**{**defaults, **overrides})


def _old_file(tmp_path: Path, name: str, age_s: float = OLD) -> Path:
    p = tmp_path / name
    p.write_bytes(b"x" * 128)
    stamp = time.time() - age_s
    os.utime(p, (stamp, stamp))
    return p


# --- age gating -------------------------------------------------------------

def test_reaps_old_matching_file(tmp_path):
    p = _old_file(tmp_path, "a.junk")
    report = reap(specs=(_spec(tmp_path),))
    assert not p.exists()
    assert report["test_debris"]["deleted"] == 1
    assert report["test_debris"]["bytes"] == 128


def test_young_file_survives(tmp_path):
    p = tmp_path / "young.junk"
    p.write_bytes(b"x")
    reap(specs=(_spec(tmp_path),))
    assert p.exists()


def test_non_matching_file_survives(tmp_path):
    p = _old_file(tmp_path, "keep.other")
    reap(specs=(_spec(tmp_path),))
    assert p.exists()


# --- liveness (lsof) ---------------------------------------------------------

def test_open_file_survives_even_when_old(tmp_path):
    p = _old_file(tmp_path, "held.junk")
    with open(p, "rb"):
        reap(specs=(_spec(tmp_path),))
        assert p.exists()  # held open by THIS process -> spared
    reap(specs=(_spec(tmp_path),))
    assert not p.exists()  # released -> reaped on the next tick


def test_open_file_survives_in_symlink_aliased_dir(tmp_path):
    # macOS reaches the Darwin temp dirs through the /var -> /private/var
    # symlink, but lsof reports RESOLVED names. Before the realpath fix this
    # aliasing made every open file look orphaned (caught live in the smoke
    # test: a held-open file was deleted on the first tick).
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    p = alias / "held.junk"
    p.write_bytes(b"x")
    stamp = time.time() - OLD
    os.utime(p, (stamp, stamp))
    with open(p, "rb"):
        reap(specs=(_spec(tmp_path, dirs=lambda: [alias]),))
        assert p.exists()


def test_open_paths_detects_held_file(tmp_path):
    held = tmp_path / "h.bin"
    held.write_bytes(b"x")
    free = tmp_path / "f.bin"
    free.write_bytes(b"x")
    with open(held, "rb"):
        live = _open_paths([held, free])
    assert held in live
    assert free not in live


# --- gates -------------------------------------------------------------------

def test_disabled_spec_deletes_nothing(tmp_path):
    p = _old_file(tmp_path, "a.junk")
    reap(specs=(_spec(tmp_path, enabled=lambda: False),))
    assert p.exists()


def test_guard_filters_per_path(tmp_path):
    doomed = _old_file(tmp_path, "doomed.junk")
    spared = _old_file(tmp_path, "spared.junk")
    reap(specs=(_spec(tmp_path, guard=lambda p: p.name.startswith("doomed")),))
    assert not doomed.exists()
    assert spared.exists()


# --- directory debris (WBPP staging) ------------------------------------------

def _old_dir(tmp_path: Path, name: str, entries: list[str]) -> Path:
    d = tmp_path / name
    d.mkdir()
    for e in entries:
        (d / e).mkdir()
    stamp = time.time() - OLD
    for p in [d, *d.iterdir()]:
        os.utime(p, (stamp, stamp))
    return d


def test_wbpp_marker_guard():
    assert _looks_like_wbpp_staging.__name__  # importable, used below via spec


def test_wbpp_staging_dir_reaped_only_with_marker(tmp_path):
    wbpp = _old_dir(tmp_path, "tmpwbpp1", ["calibrated", "logs"])
    foreign = _old_dir(tmp_path, "tmpother", ["somebodys_data"])
    spec = _spec(
        tmp_path, pattern="tmp*", guard=_looks_like_wbpp_staging
    )
    report = reap(specs=(spec,))
    assert not wbpp.exists()
    assert foreign.exists()
    assert report["test_debris"]["deleted"] == 1


def test_live_wbpp_staging_survives_via_fresh_child_mtime(tmp_path):
    # The top dir is old, but a child was written moments ago — the
    # newest-mtime rule keeps an ACTIVE run's staging alive.
    d = _old_dir(tmp_path, "tmpwbpp2", ["calibrated"])
    fresh = d / "RUN_0"
    fresh.mkdir()  # fresh mtime, not backdated
    reap(specs=(_spec(tmp_path, pattern="tmp*", guard=_looks_like_wbpp_staging),))
    assert d.exists()


# --- resilience ----------------------------------------------------------------

def test_broken_spec_never_breaks_the_tick(tmp_path):
    p = _old_file(tmp_path, "a.junk")

    def boom() -> list[Path]:
        raise RuntimeError("spec exploded")

    broken = _spec(tmp_path, name="broken", dirs=boom)
    report = reap(specs=(broken, _spec(tmp_path)))
    assert not p.exists()  # healthy spec still ran
    assert "broken" not in report
