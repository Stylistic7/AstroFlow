"""Sidecar updates are atomic read-modify-writes.

Each writer owns its own section (plate_solution, analysis, ...) and the
sections of one sidecar are written from different ops, processes and
cluster nodes. Without a lock around load -> update -> save, two writers
interleave and the later save drops the earlier section: on 2026-10-10
that lost `plate_solution` on 99 masters.
"""
import os
import threading
import time
from pathlib import Path

import pytest

from file_io import sidecar
from file_io.sidecar import get_section, set_section, sidecar_path_for


@pytest.fixture
def frame(tmp_path):
    p = tmp_path / "frame.fits"
    p.write_bytes(b"not really fits")
    return p


def _lock_path(frame: Path) -> Path:
    sc = sidecar_path_for(frame)
    return sc.with_name(sc.name + ".lock")


class TestUpdateSidecar:
    def test_concurrent_writers_keep_every_section(self, frame):
        n_writers, n_rounds = 8, 20
        errors: list[BaseException] = []

        def writer(key: str) -> None:
            try:
                for i in range(n_rounds):
                    set_section(frame, key, i)
            except BaseException as e:  # surfaced below; a thread's raise is silent
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"w{k}",)) for k in range(n_writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for k in range(n_writers):
            assert get_section(frame, f"w{k}") == n_rounds - 1, f"w{k} was dropped by another writer"
        assert not _lock_path(frame).exists()

    def test_stale_lock_is_broken(self, frame, capsys):
        lock = _lock_path(frame)
        lock.touch()
        ten_minutes_ago = time.time() - 600
        os.utime(lock, (ten_minutes_ago, ten_minutes_ago))

        set_section(frame, "analysis", {"ok": True})

        assert get_section(frame, "analysis") == {"ok": True}
        assert not lock.exists()
        assert "broken" in capsys.readouterr().out

    def test_fresh_lock_makes_the_second_writer_wait(self, frame):
        lock = _lock_path(frame)
        held = threading.Event()

        def holder() -> None:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            held.set()
            time.sleep(0.3)
            lock.unlink()

        t = threading.Thread(target=holder)
        t.start()
        held.wait()
        t0 = time.monotonic()
        set_section(frame, "plate_solution", "wcs")
        waited = time.monotonic() - t0
        t.join()

        assert waited >= 0.25, f"did not wait for the holder ({waited:.3f}s)"
        assert get_section(frame, "plate_solution") == "wcs"
        assert not lock.exists()

    def test_lock_never_released_fails_fast(self, frame, monkeypatch):
        monkeypatch.setattr(sidecar, "_LOCK_WAIT_S", 0.2)
        _lock_path(frame).touch()
        with pytest.raises(TimeoutError):
            set_section(frame, "analysis", 1)
        assert get_section(frame, "analysis") is None
