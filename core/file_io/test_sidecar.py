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


class TestEveryWriterKeepsSections:
    """2026-10-14: every path that writes a sidecar goes through the lock and
    merges, so no writer can drop another's section."""

    def test_header_update_keeps_sections(self, tmp_path):
        from file_io.generic import update_header
        from file_io.sidecar import get_section, load_sidecar, set_section, sidecar_path_for
        from file_io.types import FitsFilePath, WithSidecar
        data = tmp_path / "m.fits"; data.write_bytes(b"x")
        set_section(data, "plate_solution", {"ok": 1})
        update_header(WithSidecar(FitsFilePath(data)), {"STBLHASH": "abc"})
        update_header(WithSidecar(FitsFilePath(data)), {"FILTER": "O"})
        assert get_section(data, "plate_solution") == {"ok": 1}
        assert load_sidecar(sidecar_path_for(data))["header"] == {"STBLHASH": "abc", "FILTER": "O"}

    def test_concurrent_header_and_section_writers_keep_both(self, tmp_path):
        import threading
        from file_io.generic import update_header
        from file_io.sidecar import get_section, load_sidecar, set_section, sidecar_path_for
        from file_io.types import FitsFilePath, WithSidecar
        data = tmp_path / "m.fits"; data.write_bytes(b"x")
        def hdr(i): update_header(WithSidecar(FitsFilePath(data)), {f"K{i}": i})
        def sec(i): set_section(data, f"s{i}", i)
        ts = [threading.Thread(target=f, args=(i,)) for i in range(20) for f in (hdr, sec)]
        [t.start() for t in ts]; [t.join() for t in ts]
        root = load_sidecar(sidecar_path_for(data))
        assert all(get_section(data, f"s{i}") == i for i in range(20))
        assert all(root["header"][f"K{i}"] == i for i in range(20))

    def test_copy_sidecar_merges_into_the_destination(self, tmp_path):
        from file_io.sidecar import copy_sidecar, get_section, set_section
        src = tmp_path / "a.xisf"; dst = tmp_path / "a.fits"
        src.write_bytes(b"x"); dst.write_bytes(b"yy")
        set_section(src, "image_metadata", {"from": "src"})
        set_section(dst, "plate_solution", {"from": "dst"})
        copy_sidecar(src, dst)
        assert get_section(dst, "image_metadata") == {"from": "src"}
        assert get_section(dst, "plate_solution") == {"from": "dst"}

    def test_no_temp_files_are_left_behind(self, tmp_path):
        from file_io.sidecar import set_section
        data = tmp_path / "m.fits"; data.write_bytes(b"x")
        for i in range(5):
            set_section(data, "k", i)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["m.fits", "m.fits.json"]
