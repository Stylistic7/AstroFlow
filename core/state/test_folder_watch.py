"""`state.folder_watch`: real OS notifications in a temp folder settle into
one check, and only for files that can be data."""
import threading
import time

from state.folder_watch import FolderWatcher, meaningful


def test_sidecars_partials_and_hidden_files_are_not_data():
    assert meaningful("/raw/2026-10-22/350mm/Light_001.fits")
    assert meaningful("/raw/2026-10-22")                      # a folder moved in
    for name in ("Light_001.fits.json", "Light_001.fits.part", ".DS_Store", "x.lock"):
        assert not meaningful(f"/raw/{name}")


def _wait(event: threading.Event, seconds: float) -> bool:
    return event.wait(seconds)


def test_a_new_file_settles_into_one_check_after_the_folder_goes_quiet(tmp_path):
    calls, fired = [], threading.Event()
    watcher = FolderWatcher(lambda: (calls.append(time.monotonic()), fired.set()), poll_s=0.05)
    try:
        watcher.watch({str(tmp_path): 0.5})
        time.sleep(0.5)                                       # FSEvents needs a moment to arm
        night = tmp_path / "2026-10-22"
        night.mkdir()
        started = time.monotonic()
        for i in range(3):                                    # a batch arriving keeps resetting the clock
            (night / f"Light_{i}.fits").write_bytes(b"x" * 10)
            time.sleep(0.2)
        assert _wait(fired, 10)
        assert calls[0] - started >= 0.5 + 0.4                # quiet counted from the LAST file
        time.sleep(1.0)
        assert len(calls) == 1                                # one check per settled batch
    finally:
        watcher.stop()


def test_only_sidecars_never_wake_it(tmp_path):
    fired = threading.Event()
    watcher = FolderWatcher(fired.set, poll_s=0.05)
    try:
        watcher.watch({str(tmp_path): 0.2})
        time.sleep(0.5)
        (tmp_path / "Light_0.fits.json").write_text("{}")
        (tmp_path / ".DS_Store").write_bytes(b"")
        assert not _wait(fired, 2.0)
    finally:
        watcher.stop()


def test_an_unwatched_folder_stops_counting_and_a_missing_one_is_skipped(tmp_path):
    fired = threading.Event()
    watcher = FolderWatcher(fired.set, poll_s=0.05)
    try:
        watcher.watch({str(tmp_path): 0.2, str(tmp_path / "not-mounted"): 0.2})
        watcher.watch({})
        time.sleep(0.3)
        (tmp_path / "Light_0.fits").write_bytes(b"x")
        assert not _wait(fired, 2.0)
    finally:
        watcher.stop()
