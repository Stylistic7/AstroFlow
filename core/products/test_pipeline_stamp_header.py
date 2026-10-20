"""`products.registry` writes the pipeline stamp (`PIPEHASH`) on every
produced image: the science code, the op, and the stamp of the first image
bound to a NAMED parameter — never of a vararg (calibration masters are data)."""
import inspect
from pathlib import Path

import numpy as np

from file_io.generic import read_header
from file_io.types import Produced, TiffFilePath
from hashing.pipeline_stamp import pipeline_stamp, science_code_hash
from products import registry
from testing.shared_masters import solved_master_metadata


def _write_float_tiff(path, a):
    import pyvips
    h, w = a.shape[:2]
    pyvips.Image.new_from_memory(np.ascontiguousarray(a).tobytes(), w, h, 1, "float").tiffsave(str(path))


A = (np.arange(60 * 40, dtype="float32").reshape(40, 60) / 1e4)


def _producer(store, op):
    return registry.makes_image_wrap(
        op, filename_fn=lambda *a, **k: TiffFilePath(store / "abc123.tiff"),
        canonical_call_hash=lambda *a, **k: "abc123", makes_tiff=False, image_type=None)


def _body(out):
    _write_float_tiff(out, A)
    return Produced(out)


def test_an_op_with_no_image_input_stamps_the_science_code_and_its_name(tmp_path):
    store = tmp_path / "store"
    store.mkdir()

    def stack_files(stack, config, *cal_masters, out: TiffFilePath) -> Produced:
        return _body(out)

    upstream = solved_master_metadata(tmp_path / "flat.fits", stamp="f1a7000000000001")
    _producer(store, stack_files)("stack", "config", upstream)
    header = read_header(store / "abc123.tiff")
    assert header["PIPEHASH"] == pipeline_stamp("stack_files", None)
    assert header["CALLHASH"] == "abc123"


def test_the_primary_named_image_input_is_folded_in(tmp_path):
    store = tmp_path / "store"
    store.mkdir()

    def post_stack_calibrate(stack, config, master, *, out: TiffFilePath) -> Produced:
        return _body(out)

    master = solved_master_metadata(tmp_path / "m.fits", stamp="feed000000000001")
    _producer(store, post_stack_calibrate)("stack", "config", master)
    header = read_header(store / "abc123.tiff")
    assert header["PIPEHASH"] == pipeline_stamp("post_stack_calibrate", "feed000000000001")
    assert header["PIPEHASH"] != pipeline_stamp("post_stack_calibrate", None)


def test_the_stamp_is_a_pure_function_of_code_op_and_upstream():
    assert pipeline_stamp("x", "u") == pipeline_stamp("x", "u", code=science_code_hash())
