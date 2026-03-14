import ray
import tarfile
import tempfile
import shutil
import diskcache
import os

from context import get_cache
from metadata.image_metadata import load_metadata
from stacks.simple_stack import SimpleStack
from stacks.composite_stack import CompositeStack

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
    testdata_cache["stacks"] |= {stack_id: stacks[stack_id]}

    stack = stacks[stack_id]
    assert isinstance(stack, CompositeStack)
    for substack in stack.contents.values():
        assert isinstance(substack, SimpleStack)
        for frame_hash in substack.frames:
            key = ("compute_image_metadata", frame_hash)
            testdata_cache[key] = metadata[frame_hash]
            print(f"### Copied metadata for {frame_hash=}")


def copy_all_target_data(metadata, cache, testdata_cache, target_id):
    targets = cache["targets"]

    target = targets[target_id]
    testdata_cache["targets"] |= {target_id: target}

    for stack_id in target.stack_ids:
        copy_all_stack_data(metadata, cache, testdata_cache, stack_id)


def copy_image(metadata, cache, testdata_cache, content_hash, target_dir):
    key = ("compute_image_metadata", content_hash)
    testdata_cache[key] = metadata[content_hash]

    source_filename = metadata[content_hash].filename
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(source_filename, target_dir)
    print(f"### Copied file {source_filename} to {target_dir}")


def copy_all_mosaic_data(metadata, cache, testdata_cache, mosaic_id):
    targets = cache["targets"]
    mosaics = cache["mosaics"]

    mosaic = mosaics[mosaic_id]

    testdata_cache["mosaics"] |= {mosaic_id: mosaic}

    testdata_cache["targets"] |= {
        target_id: targets[target_id] for target_id in mosaic.target_ids
    }

    stack_ids = [
        stack_id
        for target_id in mosaic.target_ids
        for stack_id in targets[target_id].stack_ids
    ]
    for stack_id in stack_ids:
        copy_all_stack_data(metadata, cache, testdata_cache, stack_id)


if __name__ == "__main__":
    cache = get_cache()

    testdata_cache_dir = tempfile.TemporaryDirectory()
    testdata_cache = diskcache.Cache(testdata_cache_dir.name)
    testdata_cache.clear()
    testdata_cache["stacks"] = {}
    testdata_cache["targets"] = {}
    testdata_cache["mosaics"] = {}
    print(f"Opened testdata cache in dir {testdata_cache_dir.name}")

    metadata = ray.get(load_metadata.remote()).get()

    copy_all_mosaic_data(metadata, cache, testdata_cache, "75a1da6463e118e6")
    copy_all_stack_data(metadata, cache, testdata_cache, "8768e9be7664d771")
    copy_all_target_data(metadata, cache, testdata_cache, "bedcbfed642ec2b9")
    # copy_all_target_data(metadata, cache, testdata_cache, "24059224549fd764")

    # Copy the output files, and store the cache to a tar.gz file on disk.
    # shutil.copytree(context.output_dir, 'testdata', dirs_exist_ok=True)
    print(f"testdata_cache has {len(list(testdata_cache.iterkeys()))} keys")
    testdata_cache.close()
    create_tar_gz_from_directory(
        testdata_cache.directory, TESTDATA_CACHE_GZIP_FILENAME
    )
