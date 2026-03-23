import os
import shutil
import h5py
from typing import TypeVar
import ray
from pathlib import Path
import uuid
import numpy as np

from context import context
from dependency_graph import Graph, run_graph

T = TypeVar("T")


@ray.remote
def write_hdf5_file(value, path):
    image = np.full((100, 100), value, dtype=int)
    h5_filename = (path / str(uuid.uuid4())).with_suffix(".h5")
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset(f"image_{value}", data=image)
    print(f"Wrote HDF5 file {h5_filename}")
    return h5_filename


@ray.remote
def merge_hdf5_files(output_h5_filename, *input_h5_filenames):
    # Create the output HDF5 file
    with h5py.File(output_h5_filename, "w") as output_h5:
        for h5_file in input_h5_filenames:
            print(f"Merging: {h5_file}")
            with h5py.File(h5_file, "r") as input_h5:
                # Iterate over all datasets in the input file
                for key in input_h5.keys():
                    data = input_h5[key][:]

                    # Check if the dataset already exists in the output file
                    if key in output_h5:
                        raise ValueError(
                            f"Duplicate dataset key '{key}' found in file '{h5_file}'."
                        )

                    # Copy the dataset to the output file
                    output_h5.create_dataset(key, data=data)

    return output_h5_filename


@ray.remote
def rmtree(path):
    shutil.rmtree(path)


def build_graph(output_h5_filename) -> Graph:
    temp_dir = Path(context.output_dir) / str(uuid.uuid4())
    os.makedirs(temp_dir)

    return Graph(
        {
            (rmtree, temp_dir): {(merge_hdf5_files, output_h5_filename)},
            (merge_hdf5_files, output_h5_filename): {
                (write_hdf5_file, i, temp_dir) for i in range(7)
            },
        }
    )


def test_simple_graph(test_context) -> None:
    output_h5_filename = Path(context.output_dir) / "foo.h5"

    graph = build_graph(output_h5_filename)
    run_graph(graph)

    valid_range = range(7)
    with h5py.File(output_h5_filename, "r") as h5_file:
        expected_keys = {f"image_{i}" for i in valid_range}
        actual_keys = set(h5_file.keys())
        assert actual_keys == expected_keys

        for i in valid_range:
            key = f"image_{i}"
            dataset = h5_file[key]

            assert dataset.shape == (100, 100)
            assert (dataset[...] == i).all()
