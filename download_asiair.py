#!/usr/bin/env python3

import subprocess
import re
from pathlib import Path
import tempfile
import time
import datetime
import itertools
import os
import astropy.io.fits as pyfits

import file_io
import time_util

# datetime_regex = re.compile('(\d{8}-\d{6})_(\d{4})')


def ping(ip):
  return subprocess.run(["ping", "-c", "1", ip],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL).returncode == 0


def mount_volume(ip, volume, mount_point):
  ret = subprocess.run(
      ['mount', '-t', 'smbfs', f'//guest@{ip}/{volume}',
       mount_point]).returncode
  if ret != 0:
    return False

  time.sleep(5)
  return True


def unmount_volume(mount_point):
  ret = subprocess.run(['umount', mount_point]).returncode
  if ret != 0:
    return False

  time.sleep(5)
  return True


def make_dest_dir_for_date(dest_dir, date):
  dest_dir = os.path.join(dest_dir, date.strftime('%Y-%m-%d'))
  print(f"Making destination directory '{dest_dir}'")
  os.makedirs(dest_dir, exist_ok=True)

  return dest_dir


def modify_fits_headers(fits_files, rig, default_filter):
  print(f'Updating FITS headers of {len(fits_files)} files...')
  for filename in fits_files:
    assert os.path.exists(filename)
    with pyfits.open(filename, mode='update') as fits:
      dt = time_util.parse_utc_datetime(fits[0].header['DATE-OBS'])
      fits[0].header['RIG'] = rig
      fits[0].header['DATE'] = dt.strftime('%Y-%m-%d')
      if 'FILTER' not in fits[0].header:
        assert default_filter, "--default-filter must be specified"
        fits[0].header['FILTER'] = default_filter
      fits.flush()

  print('Done updating FITS headers')
  return fits_files


def copy_fits_files(fits_files, dest_dir):
  if not fits_files:
    return []

  print(f"Using rsync to copy {len(fits_files)} FITS files to '{dest_dir}'...")
  ret = subprocess.run(['rsync', '-uza', *fits_files, dest_dir]).returncode
  if ret != 0:
    print("Could not rsync FITS files")
    return

  print('rsync is done')
  return [os.path.join(dest_dir, os.path.basename(f)) for f in fits_files]


class MountContextManager:
  def __init__(self, rig, ip, volume, mount_point):
    self.rig, self.ip, self.volume, self.mount_point = rig, ip, volume, mount_point

  def __enter__(self):
    print(f"Mounting volume '{self.volume}' from rig '{self.rig}'")
    if not mount_volume(self.ip, self.volume, self.mount_point):
      print(f"Could not mount '{self.volume}' from rig '{self.rig}'")

  def __exit__(self, exc_type, exc_value, exc_tb):
    if exc_type:
      print('Exception info:', exc_type, exc_value, exc_tb, sep="\n")

    print(f"Unmounting '{self.volume}' from rig '{self.rig}'")
    if not unmount_volume(self.mount_point):
      print(f"Could not unmount volume '{self.volume}' from '{self.rig}'")


def group_subs_by_date(subs):
  def date_predicate(sub):
    utc_dt = file_io.read_fits(sub)['DATE-OBS']
    return datetime.datetime(utc_dt.year, utc_dt.month, utc_dt.day)

  groups = []
  dates = []
  print(f'group_subs_by_date found {len(subs)} frames')
  subs = sorted(subs, key=date_predicate)
  for k, g in itertools.groupby(subs, key=lambda x: date_predicate(x)):
    groups.append(list(g))
    dates.append(k)

  print(f'group_subs_by_date returning {len(groups)} groups')
  return groups, dates


def download_group(fits_files, dest_dir, rig, default_filter):
  fits_files = copy_fits_files(fits_files, dest_dir)
  fits_files = modify_fits_headers(fits_files, rig, default_filter)


def download_from_rig_grouped_by_date(path, dest_path, rig, default_filter):
  fits_files = list(Path(path).rglob('*.fit'))
  print(f'Found {len(fits_files)} FITS files from rig {rig} in path {path}')
  if not fits_files:
    return

  groups, dates = group_subs_by_date(fits_files)
  for fits_files, date in zip(groups, dates):
    date_dir = make_dest_dir_for_date(dest_path, date)
    print(f'Downloading {len(fits_files)} frames from rig {rig} '
          f'for date {date} to dir {date_dir}')
    download_group(fits_files, date_dir, rig, default_filter)

  return min(dates)


def download_logs_from_rig(rig, log_dir, dest_dir):
  if not os.path.exists(log_dir):
    print(f"Could not find log dir '{log_dir}'")
    return False

  dest_dir = os.path.join(dest_dir, f'{rig}_log')
  print(f"Copying logs from '{log_dir}' to '{dest_dir}'")
  return subprocess.run(['rsync', '-zar', log_dir, dest_dir]).returncode == 0


def download_from_rig(rig, ip, search_defs, dest_path, default_filter):
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
          min_date = download_from_rig_grouped_by_date(path, dest_path, rig,
                                                       default_filter)
          total_min_date = min(total_min_date, min_date) if (
              min_date and total_min_date) else min_date

        if total_min_date:
          log_dir = os.path.join(mount_point, search_def['log_dir'])
          date_dir = make_dest_dir_for_date(dest_path, total_min_date)
          print(f'Copying logs from source log_dir {log_dir} '
                f'to date_dir {date_dir}')
          download_logs_from_rig(rig, log_dir, date_dir)
