import datetime
import tarfile
import tempfile
import shutil
import diskcache
import os

from context import get_cache

TESTDATA_CACHE_GZIP_FILENAME = 'testdata/cache.tar.gz'


def create_tar_gz_from_directory(source_dir, output_filename):
  with tarfile.open(output_filename, "w:gz") as tar:
    for root, _, files in os.walk(source_dir):
      for file in files:
        full_path = os.path.join(root, file)
        # Add each file with a relative path within the source directory
        relative_path = os.path.relpath(full_path, start=source_dir)
        tar.add(full_path, arcname=relative_path)


def copy_testdata_into_cache():
  testdata_cache_dir = tempfile.TemporaryDirectory()

  with tarfile.open(TESTDATA_CACHE_GZIP_FILENAME, "r:gz") as tar:
    tar.extractall(path=testdata_cache_dir.name)

  testdata_cache = diskcache.Cache(testdata_cache_dir.name)
  print(f'### testdata_cache has keys {list(testdata_cache.iterkeys())}')

  cache = get_cache()
  num_copied = 0
  for key in testdata_cache.iterkeys():
    print(f'### Copied key {key} into cache')
    cache[key] = testdata_cache[key]
    num_copied += 1

  print(f'Copied {num_copied} keys from testdata cache')


def copy_all_mosaic_data(cache, testdata_cache, mosaic_id):
  mosaic = cache['grouped_mosaics'][mosaic_id]
  testdata_cache['grouped_mosaics'] = {mosaic_id: mosaic}
  testdata_cache['grouped_targets'] = {
      target_id: cache['grouped_targets'][target_id]
      for target_id in mosaic.target_ids
  }
  testdata_cache['grouped_stacks'] = {
      stack_id: cache['grouped_stacks'][stack_id]
      for target_id in mosaic.target_ids
      for stack_id in cache['grouped_targets'][target_id].stack_ids
  }

  print(f'Copied data for mosaic {mosaic_id}')


if __name__ == '__main__':
  cache = get_cache()

  testdata_cache_dir = tempfile.TemporaryDirectory()
  testdata_cache = diskcache.Cache(testdata_cache_dir.name)
  testdata_cache.clear()
  print(f'Opened testdata cache in dir {testdata_cache_dir.name}')

  copy_all_mosaic_data(cache, testdata_cache, '390cdaf04d588c91')

  # Copy the output files, and store the cache to a tar.gz file on disk.
  # shutil.copytree(context.output_dir, 'testdata', dirs_exist_ok=True)
  print(f'testdata_cache has {len(list(testdata_cache.iterkeys()))} keys')
  testdata_cache.close()
  create_tar_gz_from_directory(testdata_cache.directory,
                               TESTDATA_CACHE_GZIP_FILENAME)
