import ray
import subprocess
import re
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
import time_util

from context import context
from dependency_graph import Graph

# TODO: Improve / strengthen all the types in this file.

_SEARCH_DEFS = [{
    'volume': 'EMMC%20Images',
    'log_dir': 'log',
    'fits_paths': ['Autorun', 'Plan']
}, {
    'volume': 'TF%20Images',
    'log_dir': 'ASIAIR/log',
    'fits_paths': ['ASIAIR/Autorun', 'ASIAIR/Plan']
}]


def ping(ip: str) -> bool:
  return subprocess.run(["ping", "-c", "1", ip],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL).returncode == 0


def _mount_volume(ip: str, volume: str, mount_point: str) -> bool:
  ret = subprocess.run(
      ['mount', '-t', 'smbfs', f'//guest@{ip}/{volume}',
       mount_point]).returncode
  if ret != 0:
    return False

  time.sleep(5)
  return True


def _unmount_volume(mount_point: str) -> bool:
  ret = subprocess.run(['umount', mount_point]).returncode
  if ret != 0:
    return False

  time.sleep(5)
  return True


def _make_dest_dir_for_date(dest_dir: str, date: datetime.datetime) -> str:
  dest_dir = os.path.join(dest_dir, date.strftime('%Y-%m-%d'))
  print(f"Making destination directory '{dest_dir}'")
  os.makedirs(dest_dir, exist_ok=True)

  return dest_dir


# TODO: Remove `rig` or somehow bring it up through the UI?
def _modify_fits_headers(fits_files: list[str], rig: str,
                         default_filter: str) -> list[str]:
  print(f'Updating FITS headers of {len(fits_files)} files...')
  for filename in fits_files:
    assert os.path.exists(filename)
    fits = pyfits.open(filename, mode='update', mmap=False)
    dt = time_util.parse_utc_datetime(fits[0].header['DATE-OBS'])
    fits[0].header['RIG'] = rig
    fits[0].header['DATE'] = dt.strftime('%Y-%m-%d')
    if 'FILTER' not in fits[0].header:
      assert default_filter, "--default-filter must be specified"
      fits[0].header['FILTER'] = default_filter
    fits.flush()
    fits.close()

  print('Done updating FITS headers')
  return fits_files


def _copy_fits_files(fits_files: list[str], dest_dir: str) -> list[str] | None:
  if not fits_files:
    return []

  print(f"Using rsync to copy {len(fits_files)} FITS files to '{dest_dir}'...")
  ret = subprocess.run(['rsync', '-ua', '--inplace', *fits_files,
                        dest_dir]).returncode
  if ret != 0:
    print("Could not rsync FITS files")
    return None

  print('rsync is done')
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
      print('Exception info:', exc_type, exc_value, exc_tb, sep="\n")

    print(f"Unmounting '{self.volume}' from rig '{self.rig}'")
    if not _unmount_volume(self.mount_point):
      print(f"Could not unmount volume '{self.volume}' from '{self.rig}'")


def _group_subs_by_date(
    subs: list[str]) -> tuple[list[list[str]], list[datetime.datetime]]:

  def date_predicate(sub: str) -> datetime.datetime:
    utc_dt = file_io.fits_file_io.read_fits_header(sub)['DATE-OBS']
    return datetime.datetime(utc_dt.year, utc_dt.month, utc_dt.day)

  groups = []
  dates = []
  print(f'group_subs_by_date found {len(subs)} frames')
  sorted_subs = sorted(subs, key=date_predicate)
  for k, g in itertools.groupby(sorted_subs, key=lambda x: date_predicate(x)):
    groups.append(list(g))
    dates.append(k)

  print(f'group_subs_by_date returning {len(groups)} groups')
  return groups, dates


def _download_group(fits_files: list[str], dest_dir: str, rig: str,
                    default_filter: str) -> None:
  copied_fits_files = _copy_fits_files(fits_files, dest_dir)
  assert copied_fits_files is not None
  modified_fits_files = _modify_fits_headers(copied_fits_files, rig,
                                             default_filter)


def _download_from_rig_grouped_by_date(
    path: str, dest_path: str, rig: str,
    default_filter: str) -> datetime.datetime | None:
  fits_files = [str(f) for f in list(Path(path).rglob('*.fit'))]
  print(f'Found {len(fits_files)} FITS files from rig {rig} in path {path}')
  if not fits_files:
    return None

  groups, dates = _group_subs_by_date(fits_files)
  for fits_files, date in zip(groups, dates):
    date_dir = _make_dest_dir_for_date(dest_path, date)
    print(f'Downloading {len(fits_files)} frames from rig {rig} '
          f'for date {date} to dir {date_dir}')
    _download_group(fits_files, date_dir, rig, default_filter)

  return min(dates)


def _download_logs_from_rig(rig: str, log_dir: str, dest_dir: str) -> bool:
  if not os.path.exists(log_dir):
    print(f"Could not find log dir '{log_dir}'")
    return False

  dest_dir = os.path.join(dest_dir, f'{rig}_log')
  print(f"Copying logs from '{log_dir}' to '{dest_dir}'")
  return subprocess.run(['rsync', '-zar', log_dir, dest_dir]).returncode == 0


@ray.remote
def download_from_asiair(
    rig: str,
    ip: str,
    dest_path: str,
    default_filter: str,
    search_defs: list[dict[str, str]] = _SEARCH_DEFS) -> None:
  print(f"Pinging rig '{rig}' ({ip})")
  if not ping(ip):
    print(f"Rig '{rig}' ({ip}) is not responding.")
    return

  for search_def in search_defs:
    with tempfile.TemporaryDirectory() as mount_point:
      with MountContextManager(rig, ip, search_def['volume'], mount_point):
        print(f'Successfully mounted rig {rig} ({ip}) '
              f'at mount point {mount_point}')

        total_min_date = None
        for fits_path in search_def['fits_paths']:
          path = os.path.join(mount_point, fits_path)
          print(f'Copying FITS files from path {path}')
          min_date = _download_from_rig_grouped_by_date(path, dest_path, rig,
                                                        default_filter)
          if total_min_date is None:
            total_min_date = min_date
          else:
            total_min_date = min(total_min_date, min_date)

        if total_min_date:
          log_dir = os.path.join(mount_point, search_def['log_dir'])
          date_dir = _make_dest_dir_for_date(dest_path, total_min_date)
          print(f'Copying logs from source log_dir {log_dir} '
                f'to date_dir {date_dir}')
          _download_logs_from_rig(rig, log_dir, date_dir)


@ray.remote
def downloads_complete():
  pass

@ray.remote
def download_from_asiairs() -> Graph:
  return Graph({
      downloads_complete: {(download_from_asiair, 'foo', dev['ip'],
                            context.input_dirs[0], 'H')
                           for dev in context.asiair_devices}
  })
