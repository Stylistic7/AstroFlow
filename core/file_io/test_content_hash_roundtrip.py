"""A produced image's content_hash must round-trip through its header stamp —
even when the file was read (and libvips-cached) before it was stamped.

Regression for the "re-derives all night" churn: `update_tiff_header` opens the
file to merge metadata (populating libvips's operation cache with the *pre-stamp*
image), then overwrites it in place with the CONTHASH stamp. Without
`revalidate=True` on reads, a subsequent same-process read returned the STALE,
un-stamped header, so `get_content_hash` fell back to **pixel-hashing an output
file** — giving produced images a non-deterministic identity that never
cache-hit, so `merge_images_oss` re-ran every trace and the graph never
converged. A produced (output-dir) file must always take its content_hash from
the stamped call-hash; only raw subframes are ever pixel-hashed.
"""
import numpy as np
import pytest

from file_io.tiff_file_io import write_tiff_image, read_tiff_header
from file_io.generic import set_content_hash, get_content_hash, hash_of_image_data
from file_io.types import TiffFilePath


@pytest.mark.parametrize("shape", [(300, 300), (200, 220, 3)])
def test_content_hash_stamp_survives_pyvips_cache(tmp_path, shape):
    p = TiffFilePath(tmp_path / "produced.tiff")
    write_tiff_image(p, np.random.default_rng(0).random(shape, dtype=np.float32),
                     {"FILTER": "H"})

    # Populate libvips's operation cache with the PRE-stamp image (the trap that
    # made this fail): any read before the stamp used to poison later reads.
    read_tiff_header(p)

    # makes_image_wrap stamps the deterministic call-hash as the content_hash.
    set_content_hash(p, "call_hash_deadbeef")

    got = get_content_hash(p)
    assert got == "call_hash_deadbeef", (
        f"produced image content_hash fell back to pixel-hashing: {got} "
        f"(pixel-hash={hash_of_image_data(p)}) — the stamp was read stale from "
        f"libvips's cache. Reads must revalidate.")
    assert read_tiff_header(p).get("CONTHASH") == "call_hash_deadbeef"


def test_all_digit_content_hash_round_trips_as_a_string(tmp_path):
    """An all-digit call hash written to an XISF comes back through the
    reader's `literal_eval` as an int; `get_content_hash` is an identifier
    and must return the string (boot 91: `4526235228981703 ==
    '4526235228981703'` was False, so one master was re-stacked every pass)."""
    from file_io.types import XisfFilePath
    from file_io.xisf_file_io import write_xisf_image
    p = XisfFilePath(tmp_path / "m.xisf")
    write_xisf_image(p, np.zeros((8, 8, 1), np.float32), {"CONTHASH": "4526235228981703"})
    h = get_content_hash(p)
    assert h == "4526235228981703" and isinstance(h, str)
