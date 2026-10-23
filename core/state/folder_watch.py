"""Watching the raw data folders and the shared master folders.

The operating system says when something under a folder changes (FSEvents
on macOS, inotify on Linux, ReadDirectoryChanges on Windows, through
`watchdog`); a folder that has then been quiet for its settling time is
compared with its last scan, and a real difference drains the graph and
queues a pass — exactly what the hourly check does, an hour sooner.

Two rules keep it from starting passes nobody needs:
* QUIET first: a copy or a sync in progress keeps resetting the folder's
  clock, so a pass never starts on half a batch. A shared folder waits out
  `shared_masters.SETTLED_AFTER_S`, so the files that woke it are old
  enough to be measured by the pass they start.
* A DIFFERENCE, not an event: AstroFlow's own sidecars, a file the ASIAir
  download already handed to a pass, a touch that changed nothing — all
  leave the folder equal to its last scan, and nothing happens.

Changes made by another machine on a network share raise no event here;
the hourly check (`folders_changed` too) is the fallback for those.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import structlog

log = structlog.get_logger()

RAW_QUIET_S = 120.0

#: Names that never mean new data: sidecars, partial downloads, Finder litter.
_IGNORED_SUFFIXES = (".json", ".part", ".tmp", ".lock")


def meaningful(path: str) -> bool:
    name = Path(path).name
    return not (name.startswith(".") or name.lower().endswith(_IGNORED_SUFFIXES))


def raw_dirs_changed(ctx) -> bool:
    """Whether any reachable raw data folder differs from its last scan."""
    from metadata.library_scan import dir_available, load_record, scan_dir
    for folder in (ctx.input_dirs or []):
        if not dir_available(folder):
            continue
        known = load_record(folder)
        if known is None or known.fingerprint != scan_dir(folder).fingerprint:
            return True
    return False


def folders_changed(ctx) -> bool:
    from stacks.publish_shared import shared_dirs_changed
    return raw_dirs_changed(ctx) or bool(ctx.shared_dirs and shared_dirs_changed(ctx))


class FolderWatcher:
    """Recursive OS watches on `folders` ({path: quiet_s}); `on_settled()`
    runs (on the watcher's own thread) once every folder that saw a
    meaningful event has been quiet for its time."""

    def __init__(self, on_settled, poll_s: float = 5.0):
        from watchdog.observers import Observer
        self._on_settled = on_settled
        self._poll_s = poll_s
        self._observer = Observer()
        self._watches: dict[str, tuple] = {}      # folder -> (watch, quiet_s)
        self._last_event: dict[str, float] = {}   # folder -> monotonic time of its latest event
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observer.start()
        self._thread = threading.Thread(target=self._loop, name="folder-watch", daemon=True)
        self._thread.start()

    def watch(self, folders: dict[str, float]) -> None:
        """Watch exactly `folders` from now on; a folder that is missing
        (an unmounted volume) is skipped until a later call finds it."""
        from watchdog.events import FileSystemEventHandler

        watcher = self

        class _Handler(FileSystemEventHandler):
            def __init__(self, folder):
                self.folder = folder

            def on_any_event(self, event):
                # A folder's "modified" only echoes a child's own event (a
                # sidecar written into it would otherwise count); opens and
                # closes change nothing.
                if event.event_type in ("opened", "closed", "closed_no_write") or (
                        event.is_directory and event.event_type == "modified") or (
                        Path(event.src_path) == Path(self.folder)):
                    return
                paths = (event.src_path, getattr(event, "dest_path", "") or "")
                if any(p and meaningful(p) for p in paths):
                    with watcher._lock:
                        watcher._last_event[self.folder] = time.monotonic()

        with self._lock:
            for folder in set(self._watches) - set(folders):
                self._observer.unschedule(self._watches.pop(folder)[0])
                self._last_event.pop(folder, None)
            for folder, quiet_s in folders.items():
                if folder in self._watches:
                    self._watches[folder] = (self._watches[folder][0], quiet_s)
                elif Path(folder).is_dir():
                    watch = self._observer.schedule(_Handler(folder), folder, recursive=True)
                    self._watches[folder] = (watch, quiet_s)
                    log.info("watching folder", folder=folder, quiet_s=quiet_s)

    def settled(self, now: float) -> bool:
        """Some folder saw events and every folder with events is quiet."""
        with self._lock:
            if not self._last_event:
                return False
            if any(now - t < self._watches[f][1] for f, t in self._last_event.items()):
                return False
            self._last_event.clear()
            return True

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_s):
            if self.settled(time.monotonic()):
                try:
                    self._on_settled()
                except Exception as e:     # never let one bad check kill the watcher
                    log.warning("folder watch check raised", error=str(e))

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=5)
