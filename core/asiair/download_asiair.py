from __future__ import annotations

import ray
import subprocess
from pathlib import Path
import tempfile
import time
import datetime
import itertools
import os
import astropy.io.fits as pyfits  # type: ignore
from typing import Any

import file_io
import file_io.fits_file_io
from file_io.sidecar import update_sidecar
import time_util

# TODO: Improve / strengthen all the types in this file.

_SEARCH_DEFS = [
    {"volume": "EMMC%20Images", "log_dir": "log", "fits_paths": ["Autorun", "Plan"]},
    {
        "volume": "TF%20Images",
        "log_dir": "ASIAIR/log",
        "fits_paths": ["ASIAIR/Autorun", "ASIAIR/Plan"],
    },
]

# Rig nicknames keyed by FOCALLEN (mm, rounded) read from the frames
# themselves — the one identity signal that survives firmware updates,
# DHCP shuffles, extender MAC translation and mDNS boot-order renaming.
# The ASIAIR's own nickname stopped being queryable at firmware 14.39
# (get_device_state no longer answers; see asiair/rpc_client.py).
# Update when a camera moves to a different scope.
RIG_NAMES_BY_FOCAL_LENGTH = {351: "RC71", 1234: "FLT156"}


def _rig_name_for_group(fits_files: list[str], fallback: str) -> str:
    """Nickname for the rig that shot these frames, from the first
    readable FOCALLEN. Frames in one group all come from one device's
    card, so one header speaks for the group."""
    for f in fits_files:
        try:
            focal = float(
                file_io.fits_file_io.read_fits_header(f)["FOCALLEN"]
            )
        except Exception:
            continue
        name = RIG_NAMES_BY_FOCAL_LENGTH.get(round(focal))
        if name:
            return name
        print(f"### unmapped FOCALLEN {focal} — filing under '{fallback}'")
        return fallback
    return fallback


def ping(ip: str) -> bool:
    return (
        subprocess.run(
            ["ping", "-c", "1", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _mount_volume(ip: str, volume: str, mount_point: str) -> bool:
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


# TODO: Remove `rig` or somehow bring it up through the UI?
def _save_fits_overlay(
    fits_files: list[str], rig: str, default_filter: str
) -> list[str]:
    """
    Read raw FITS headers and persist an AstroFlow header overlay into the sidecar.
    The FITS files themselves are never modified.
    """
    print(f"Recording FITS header overlay for {len(fits_files)} files...")

    for filename in fits_files:
        path = Path(filename)
        assert path.exists()

        # Read-only: inspect raw header.
        with pyfits.open(path, mode="readonly", memmap=True) as fits:
            hdr = fits[0].header
            date_obs = hdr.get("DATE-OBS")
            has_filter = "FILTER" in hdr

        if not date_obs:
            raise KeyError(f"{path}: missing DATE-OBS")

        dt = time_util.parse_utc_datetime(date_obs)

        def _upd(root: dict) -> None:
            astroflow = root.setdefault("astroflow", {})
            overlay = astroflow.setdefault("header_overlay", {})

            overlay["RIG"] = rig
            overlay["DATE"] = dt.strftime("%Y-%m-%d")

            if not has_filter:
                assert default_filter, "--default-filter must be specified"
                overlay["FILTER"] = default_filter
            else:
                # If raw FITS has FILTER, do not override it via overlay.
                overlay.pop("FILTER", None)

        update_sidecar(path, _upd)

    print("Done recording FITS header overlay")
    return fits_files


def _copy_fits_files(fits_files: list[str], dest_dir: str) -> list[str] | None:
    if not fits_files:
        return []

    print(f"Using rsync to copy {len(fits_files)} FITS files to '{dest_dir}'...")
    ret = subprocess.run(
        [
            "rsync",
            "-a",
            "--ignore-existing",
            "--partial",
            "--partial-dir=.rsync-partial",
            "--delay-updates",
            *fits_files,
            dest_dir,
        ]
    ).returncode
    if ret != 0:
        print("Could not rsync FITS files")
        return None

    print("rsync is done")
    return [os.path.join(dest_dir, os.path.basename(f)) for f in fits_files]


class MountContextManager:
    def __init__(self, rig: str, ip: str, volume: str, mount_point: str) -> None:
        self.rig, self.ip, self.volume, self.mount_point = rig, ip, volume, mount_point

    def __enter__(self) -> None:
        print(f"Mounting volume '{self.volume}' from rig '{self.rig}'")
        if not _mount_volume(self.ip, self.volume, self.mount_point):
            print(f"Could not mount '{self.volume}' from rig '{self.rig}'")

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:
        if exc_type:
            print("Exception info:", exc_type, exc_value, exc_tb, sep="\n")

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


def _download_group(
    fits_files: list[str], dest_dir: str, rig: str, default_filter: str
) -> None:
    copied_fits_files = _copy_fits_files(fits_files, dest_dir)
    assert copied_fits_files is not None
    _save_fits_overlay(copied_fits_files, rig, default_filter)


def _download_from_rig_grouped_by_date(
    path: str, dest_path: str, rig: str, default_filter: str
) -> tuple[datetime.datetime, str] | None:
    fits_files = [str(f) for f in list(Path(path).rglob("*.fit"))]
    print(f"Found {len(fits_files)} FITS files from rig {rig} in path {path}")
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
        _download_group(fits_files, date_dir, rig_name, default_filter)

    return min(dates), rig_name


def _download_logs_from_rig(log_dir: str, dest_dir: str) -> bool:
    """Copy the device's log tree into <date>/<rig>/logs."""
    if not os.path.exists(log_dir):
        print(f"Could not find log dir '{log_dir}'")
        return False

    dest_dir = os.path.join(dest_dir, "logs")
    print(f"Copying logs from '{log_dir}' to '{dest_dir}'")
    return subprocess.run(["rsync", "-zar", log_dir, dest_dir]).returncode == 0


@ray.remote
def download_from_asiair(
    rig: str,
    ip: str,
    dest_path: str,
    default_filter: str,
    search_defs: list[dict[str, str]] = _SEARCH_DEFS,
) -> None:
    print(f"Pinging rig '{rig}' ({ip})")
    if not ping(ip):
        print(f"Rig '{rig}' ({ip}) is not responding.")
        return

    for search_def in search_defs:
        with tempfile.TemporaryDirectory() as mount_point:
            with MountContextManager(rig, ip, search_def["volume"], mount_point):
                print(
                    f"Successfully mounted rig {rig} ({ip}) "
                    f"at mount point {mount_point}"
                )

                total_min_date = None
                rig_name = rig
                for fits_path in search_def["fits_paths"]:
                    path = os.path.join(mount_point, fits_path)
                    print(f"Copying FITS files from path {path}")
                    result = _download_from_rig_grouped_by_date(
                        path, dest_path, rig, default_filter
                    )
                    if result is None:
                        continue
                    min_date, rig_name = result
                    total_min_date = (
                        min_date if total_min_date is None
                        else min(total_min_date, min_date)
                    )

                if total_min_date:
                    log_dir = os.path.join(mount_point, search_def["log_dir"])
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


