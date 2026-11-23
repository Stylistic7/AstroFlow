import datetime
import numpy as np
import os
import pytest
import tempfile
import unittest
from pathlib import Path

from file_io.types import FitsFilePath, XisfFilePath, WithSidecar
from file_io.generic import (
    read_header,
    read_image,
    update_header,
    get_content_hash,
    set_content_hash,
    copy_image,
)
from file_io.xisf_file_io import write_xisf_image


def write_master(path: Path) -> XisfFilePath:
    """A master light as the pipeline writes one: float32, with the header
    cards the reader types (IMAGETYP, EXPTIME, DATE-OBS). Twelve by nineteen
    pixels stand in for a 12772 x 19148 stack; the reader is size-blind, and
    the 979 MB real master this file used to carry bought nothing."""
    pixels = np.random.default_rng(0).random((12, 19, 1), dtype=np.float32)
    write_xisf_image(path, pixels, {"IMAGETYP": "master light", "EXPTIME": 300.0,
                                    "DATE-OBS": "2024-02-05T17:29:43", "FILTER": "H"})
    return XisfFilePath(path)


class TestFileIO(unittest.TestCase):
    @pytest.mark.testdata
    def test_read_fits(self):
        filename = FitsFilePath(
            "testdata/input/Light_Orion_Pane 1_300.0s_Bin1_6200MM_H_20231222-210506_0001.fit"
        )
        header = read_header(filename)

        self.assertEqual(type(header["IMAGETYP"]), str)
        self.assertEqual(header["NAXIS"], 2)
        self.assertEqual(header["NAXIS1"], 9576)
        self.assertEqual(header["NAXIS2"], 6388)
        self.assertEqual(type(header["EXPTIME"]), float)
        self.assertEqual(type(header["DATE-OBS"]), datetime.datetime)

        image = read_image(filename)
        self.assertEqual(image.shape, (6388, 9576))
        self.assertEqual(image.dtype, np.uint16)

    def test_read_xisf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = write_master(Path(temp_dir) / "master.xisf")
            header = read_header(filename)

            self.assertEqual(type(header["IMAGETYP"]), str)
            self.assertEqual(header["NAXIS"], 2)
            self.assertEqual(header["NAXIS1"], 19)
            self.assertEqual(header["NAXIS2"], 12)
            self.assertEqual(type(header["EXPTIME"]), float)
            self.assertEqual(type(header["DATE-OBS"]), datetime.datetime)

            image = read_image(filename)
            self.assertEqual(image.shape, (12, 19))
            self.assertEqual(image.dtype, np.float32)

    def test_add_xisf_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_filename = write_master(Path(temp_dir) / "foo.xisf")
            update_header(output_filename, {"CONTHASH": "foo", "FLOATVAL": 1.234})
            header = read_header(output_filename)
            self.assertEqual(header["CONTHASH"], "foo")
            self.assertEqual(header["FLOATVAL"], 1.234)

    def test_set_content_hash_lands_in_native_header(self):
        """`set_content_hash` for `WithSidecar` dispatches through to
        the wrapped Path and lands in the XISF native header — single
        source of truth, no companion sidecar file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = WithSidecar(write_master(Path(temp_dir) / "foo.xisf"))
            set_content_hash(output_file, "foo")

            update_header(output_file, {"FLOATVAL": 1.234})
            header = read_header(output_file)
            self.assertEqual(header["FLOATVAL"], 1.234)

            self.assertEqual(get_content_hash(output_file), "foo")
            # Header carries the content_hash directly; get_content_hash
            # reads it without falling back to pixel-rehashing.
            self.assertEqual(read_header(output_file.path)["CONTHASH"], "foo")

    def test_copy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_file = WithSidecar(write_master(Path(temp_dir) / "foo.xisf"))
            set_content_hash(first_file, "foo")
            update_header(first_file, {"FLOATVAL": 1.234})

            second_file = WithSidecar(XisfFilePath(os.path.join(temp_dir, "bar.xisf")))
            copy_image(first_file, second_file)

            header = read_header(second_file)
            self.assertEqual(header["FLOATVAL"], 1.234)


if __name__ == "__main__":
    unittest.main()
