import ray
import tarfile
import tempfile
import shutil
import diskcache
import os

from ray_infra.ray_init import connect_to_ray_cluster
connect_to_ray_cluster()

from context import get_cache
from cache.context_ref import ContextRef
from metadata.image_metadata import load_library_now
from stacks.run import Run
from stacks.composite_stack import Stack


def _resolve(obj):
    """Resolve a ContextRef, or return the object as-is."""
    if isinstance(obj, ContextRef):
        return obj.get()
    return obj

TESTDATA_CACHE_GZIP_FILENAME = "testdata/cache.tar.gz"


def create_tar_gz_from_directory(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        for root, _, files in os.walk(source_dir):
            for file in files:
                full_path = os.path.join(root, file)
                # Add each file with a relative path within the source directory
                relative_path = os.path.relpath(full_path, start=source_dir)
                tar.add(full_path, arcname=relative_path)


def copy_all_stack_data(metadata, cache, testdata_cache, stack_id):
    stacks = cache["stacks"]
    stack = _resolve(stacks[stack_id])
    assert isinstance(stack, Stack)
    for substack in stack.contents.values():
        assert isinstance(substack, Run)
        for frame_hash in substack.frames:
            key = ("compute_image_metadata", frame_hash)
            testdata_cache[key] = metadata[frame_hash]
            print(f"### Copied metadata for {frame_hash=}")


def copy_all_target_data(metadata, cache, testdata_cache, target_id):
    targets = cache["targets"]
    target = _resolve(targets[target_id])
    for stack_id in target.stack_ids:
        copy_all_stack_data(metadata, cache, testdata_cache, stack_id)


def copy_all_mosaic_data(metadata, cache, testdata_cache, mosaic_id):
    targets = cache["targets"]
    mosaics = cache["mosaics"]
    mosaic = _resolve(mosaics[mosaic_id])
    stack_ids = [
        stack_id
        for target_id in mosaic.target_ids
        for stack_id in _resolve(targets[target_id]).stack_ids
    ]
    for stack_id in stack_ids:
        copy_all_stack_data(metadata, cache, testdata_cache, stack_id)


if __name__ == "__main__":
    cache = get_cache()

    testdata_cache_dir = tempfile.TemporaryDirectory()
    testdata_cache = diskcache.Cache(testdata_cache_dir.name)
    testdata_cache.clear()
    print(f"Opened testdata cache in dir {testdata_cache_dir.name}")

    metadata = ray.get(load_library_now.remote()).get()

    copy_all_mosaic_data(metadata, cache, testdata_cache, "75a1da6463e118e6")
    copy_all_stack_data(metadata, cache, testdata_cache, "8768e9be7664d771")
    copy_all_target_data(metadata, cache, testdata_cache, "bedcbfed642ec2b9")

    # Copy the output files, and store the cache to a tar.gz file on disk.
    # shutil.copytree(context.output_dir, 'testdata', dirs_exist_ok=True)
    print(f"testdata_cache has {len(list(testdata_cache.iterkeys()))} keys")
    testdata_cache.close()
    create_tar_gz_from_directory(
        testdata_cache.directory, TESTDATA_CACHE_GZIP_FILENAME
    )
