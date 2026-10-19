from __future__ import annotations

import ray
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote
import tempfile
import time
import datetime
import itertools
import os
from typing import Any

import file_io
import file_io.fits_file_io

# TODO: Improve / strengthen all the types in this file.

_SEARCH_DEFS = [
    {"volume": "EMMC%20Images", "log_dir": "log", "fits_paths": ["Autorun", "Plan"]},
    {
        "volume": "TF%20Images",
        "log_dir": "ASIAIR/log",
        "fits_paths": ["ASIAIR/Autorun", "ASIAIR/Plan"],
    },
]

def _rig_name_for_group(fits_files: list[str], fallback: str) -> str:
    """Per-rig folder name from the frames' own FOCALLEN, rounded to the
    nearest 10 mm — e.g. "1230mm", "350mm" (user decision 2026-09-05:
    no nicknames; the ASIAIR's nickname stopped being queryable at
    firmware 14.39, and a nickname map needs maintenance while the
    focal length is self-describing and drift-proof by rounding).
    Frames in one group all come from one device's card, so one header
    speaks for the group. FOCALLEN is the one identity signal that
    survives firmware updates, DHCP shuffles, extender MAC translation
    and mDNS boot-order renaming."""
    for f in fits_files:
        try:
            focal = float(
                file_io.fits_file_io.read_fits_header(f)["FOCALLEN"]
            )
        except Exception:
            continue
        return f"{int(round(focal / 10.0) * 10)}mm"
    return fallback


def ping(ip: str) -> bool:
    return (
        subprocess.run(
            ["ping", "-n" if sys.platform == "win32" else "-c", "1", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _unmount_stale_share(ip: str, volume: str) -> None:
    """Sweep any existing mount of this share before mounting fresh.

    A pull interrupted mid-mount (deploy restart, crash) leaves the
    share attached to a dead temp dir; macOS then refuses to mount the
    same share again ("File exists") and the pass scans an empty
    directory, reporting a healthy rig as having 0 files (2026-09-05:
    RC71's whole night looked absent)."""
    out = subprocess.run(["mount"], capture_output=True, text=True).stdout
    needle = f"@{ip}/{volume} on "
    for line in out.splitlines():
        if needle not in line:
            continue
        mount_point = line.split(" on ", 1)[1].split(" (")[0]
        print(f"### unmounting stale share {ip}/{volume} at {mount_point}")
        if subprocess.run(["umount", mount_point]).returncode != 0:
            subprocess.run(["umount", "-f", mount_point])


def _mount_volume(ip: str, volume: str, mount_point: str) -> bool:
    _unmount_stale_share(ip, volume)
    ret = subprocess.run(
        ["mount", "-t", "smbfs", f"//guest@{ip}/{volume}", mount_point]
    ).returncode
    if ret != 0:
        return False

    time.sleep(5)
    return True


def _unmount_volume(mount_point: str) -> bool:
    ret = subprocess.run(["umount", mount_point]).returncode
    if ret != 0:
        return False

    time.sleep(5)
    return True


def _make_dest_dir_for_date(dest_dir: str, date: datetime.datetime) -> str:
    dest_dir = os.path.join(dest_dir, date.strftime("%Y-%m-%d"))
    print(f"Making destination directory '{dest_dir}'")
    os.makedirs(dest_dir, exist_ok=True)

    return dest_dir


def _copy_fits_files(fits_files: list[str], dest_dir: str) -> bool:
    """Copy the candidates into `dest_dir`; True when every copy succeeded.
    A name already there is left alone; each file lands whole (written
    beside as `.partial`, then renamed) with its mtime kept — what rsync's
    `--ignore-existing --partial --delay-updates` did before, on the Mac
    only (Windows has no rsync).

    Deliberately does NOT report which files it copied. The copied set is a
    subset of the candidates, and the old return — the candidate list,
    restated with the destination prefix — was that subset's
    plausible-looking impostor. Its one consumer fed it to a sidecar rewrite
    that bumped 171 mtimes per sweep and spun the graph forever
    (2026-09-09). Nothing needs the list: the library scan discovers what
    landed, from the disk.
    """
    if not fits_files:
        return True

    print(f"Copying {len(fits_files)} FITS files to '{dest_dir}'...")
    try:
        for src in fits_files:
            dest = Path(dest_dir) / Path(src).name
            if dest.exists():
                continue
            partial = dest.with_name(dest.name + ".partial")
            shutil.copy2(src, partial)
            os.replace(partial, dest)
    except OSError as e:
        print(f"Could not copy FITS files: {e}")
        return False

    print("copy is done")
    return True


class MountContextManager:
    """The rig's share, readable at `path` inside the block: mounted at
    `mount_point` on the Mac; on Windows reached as a UNC path, no mount."""

    def __init__(self, rig: str, ip: str, volume: str, mount_point: str) -> None:
        self.rig, self.ip, self.volume, self.mount_point = rig, ip, volume, mount_point
        self.mounted = False
        self.path = mount_point

    def __enter__(self) -> "MountContextManager":
        if sys.platform == "win32":
            self.path = rf"\\{self.ip}\{unquote(self.volume)}"
            self.mounted = os.path.isdir(self.path)
            if not self.mounted:
                # The ASIAIR's share is a guest share, which Windows blocks
                # by default (AllowInsecureGuestAuth) and, since 11 24H2,
                # by requiring SMB signing the device does not do.
                print(f"Could not reach {self.path}: Windows allows a guest share only with "
                      "AllowInsecureGuestAuth on and SMB signing not required")
            return self
        print(f"Mounting volume '{self.volume}' from rig '{self.rig}'")
        self.mounted = _mount_volume(self.ip, self.volume, self.mount_point)
        if not self.mounted:
            print(f"Could not mount '{self.volume}' from rig '{self.rig}'")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        if exc_type:
            print("Exception info:", exc_type, exc_value, exc_tb, sep="\n")

        if not self.mounted or sys.platform == "win32":
            return
        print(f"Unmounting '{self.volume}' from rig '{self.rig}'")
        if not _unmount_volume(self.mount_point):
            print(f"Could not unmount volume '{self.volume}' from '{self.rig}'")


def _group_subs_by_date(
    subs: list[str],
) -> tuple[list[list[str]], list[datetime.datetime]]:
    def sub_date(sub: str) -> datetime.datetime:
        utc_dt = file_io.fits_file_io.read_fits_header(sub)["DATE-OBS"]
        return datetime.datetime(utc_dt.year, utc_dt.month, utc_dt.day)

    print(f"group_subs_by_date found {len(subs)} frames")

    # The ASIAir SMB mount routinely contains *.fit files that aren't
    # readable FITS yet: in-progress/truncated captures, empty files,
    # or non-light files. A single bad file must NOT abort the whole
    # download (it used to, inside sorted()'s key) — skip it loudly;
    # an in-progress sub is picked up on the next poll once complete.
    # Compute each date exactly once (the old code re-read every
    # header in both sorted() and groupby()).
    dated: list[tuple[str, datetime.datetime]] = []
    skipped = 0
    for sub in subs:
        try:
            dated.append((sub, sub_date(sub)))
        except Exception as e:
            skipped += 1
            print(f"### group_subs_by_date: skipping unreadable {sub}: {e}")
    if skipped:
        print(f"group_subs_by_date skipped {skipped} unreadable frame(s)")

    dated.sort(key=lambda sd: sd[1])
    groups = []
    dates = []
    for k, g in itertools.groupby(dated, key=lambda sd: sd[1]):
        groups.append([sub for sub, _ in g])
        dates.append(k)

    print(f"group_subs_by_date returning {len(groups)} groups")
    return groups, dates


def _download_from_rig_grouped_by_date(
    path: str, dest_path: str, rig: str
) -> tuple[datetime.datetime, str] | None:
    fits_files = [str(f) for f in list(Path(path).rglob("*.fit"))]
    print(f"Found {len(fits_files)} FITS files from rig {rig} in path {path}")
    if not fits_files:
        return None

    # Never copy a file the camera is still writing: it lands truncated
    # and --ignore-existing then refuses to repair it forever (lost a
    # Pane 11 H sub 2026-09-05). Data-relative rule — skip anything
    # within 60s of the CARD'S newest mtime — so device clock skew is
    # irrelevant, and once imaging stops nothing is fresh relative to
    # the newest file and the final frame always copies.
    mtimes = {}
    for f in fits_files:
        try:
            mtimes[f] = os.path.getmtime(f)
        except OSError:
            print(f"### skipping unstattable file {f}")
    if not mtimes:
        return None
    newest_mtime = max(mtimes.values())
    # The newest file is 0s from itself, so the data-relative test alone
    # would defer the final frame forever; the wall-clock backstop (5 min,
    # far beyond any write duration or sane device clock skew) copies it
    # once the card has been quiet.
    now = time.time()
    settled = [
        f for f, t in mtimes.items()
        if newest_mtime - t >= 60.0 or now - t >= 300.0
    ]
    skipped = len(fits_files) - len(settled)
    if skipped:
        print(f"### deferring {skipped} still-writing file(s) to the next poll")
    fits_files = settled
    if not fits_files:
        return None

    groups, dates = _group_subs_by_date(fits_files)
    if not dates:
        print(f"No readable FITS frames from rig {rig} in path {path}")
        return None
    rig_name = rig
    for fits_files, date in zip(groups, dates):
        # Each observing date gets one subfolder per rig, named by the
        # nickname derived from the frames' FOCALLEN:
        # <dest>/<YYYY-MM-DD>/<RC71|FLT156>/...
        rig_name = _rig_name_for_group(fits_files, rig)
        date_dir = os.path.join(
            _make_dest_dir_for_date(dest_path, date), rig_name
        )
        os.makedirs(date_dir, exist_ok=True)
        print(
            f"Downloading {len(fits_files)} frames from rig {rig_name} "
            f"for date {date} to dir {date_dir}"
        )
        assert _copy_fits_files(fits_files, date_dir)

    return min(dates), rig_name


def _download_logs_from_rig(log_dir: str, dest_dir: str) -> bool:
    """Copy the device's log tree into <date>/<rig>/logs."""
    if not os.path.exists(log_dir):
        print(f"Could not find log dir '{log_dir}'")
        return False

    dest = os.path.join(dest_dir, "logs", os.path.basename(os.path.normpath(log_dir)))
    print(f"Copying logs from '{log_dir}' to '{dest}'")
    try:
        shutil.copytree(log_dir, dest, dirs_exist_ok=True)
    except OSError as e:   # shutil.Error too: files the device is still writing
        print(f"Could not copy logs: {e}")
        return False
    return True


@ray.remote
def download_from_asiair(
    rig: str,
    ip: str,
    dest_path: str,
    *,
    search_defs: list[dict[str, str]] = _SEARCH_DEFS,
) -> None:
    """Download one rig's frames. `search_defs` is KEYWORD-ONLY on purpose.

    It used to sit fourth positionally behind `default_filter: str`. When
    `default_filter` was deleted (the header-overlay removal), the scheduler's
    call site kept passing its `"H"` — which slid silently into `search_defs`,
    iterated as the single character "H", and died on `search_def["volume"]`.
    Every ASIAir download failed, and because the poller does `ray.get(refs)`
    before its drain-and-restart, the COMPUTE GRAPH never got kicked either
    (boot 67, 2026-09-09). Keyword-only makes a stray positional a loud
    TypeError on the driver instead of a string that looks iterable.
    """
    print(f"Pinging rig '{rig}' ({ip})")
    if not ping(ip):
        print(f"Rig '{rig}' ({ip}) is not responding.")
        return

    for search_def in search_defs:
        with tempfile.TemporaryDirectory() as mount_point:
            with MountContextManager(rig, ip, search_def["volume"], mount_point) as mcm:
                if not mcm.mounted:
                    # Scanning an unmounted temp dir "finds" 0 files and
                    # reports a healthy rig as empty — skip loudly instead.
                    print(
                        f"Skipping volume {search_def['volume']} on rig "
                        f"{rig} ({ip}) — mount failed"
                    )
                    continue
                print(f"Reading rig {rig} ({ip}) at {mcm.path}")

                total_min_date = None
                rig_name = rig
                for fits_path in search_def["fits_paths"]:
                    path = os.path.join(mcm.path, fits_path)
                    print(f"Copying FITS files from path {path}")
                    result = _download_from_rig_grouped_by_date(
                        path, dest_path, rig
                    )
                    if result is None:
                        continue
                    min_date, rig_name = result
                    total_min_date = (
                        min_date if total_min_date is None
                        else min(total_min_date, min_date)
                    )

                if total_min_date:
                    log_dir = os.path.join(mcm.path, search_def["log_dir"])
                    date_dir = os.path.join(
                        _make_dest_dir_for_date(dest_path, total_min_date),
                        rig_name,
                    )
                    os.makedirs(date_dir, exist_ok=True)
                    print(
                        f"Copying logs from source log_dir {log_dir} "
                        f"to date_dir {date_dir}"
                    )
                    _download_logs_from_rig(log_dir, date_dir)


