

def test_get_content_hash_never_decodes_a_stamped_file(tmp_path, monkeypatch):
    """dict.get(KEY, hash_of_image_data(f)) evaluated the fallback EAGERLY —
    every stamped file was full-pixel-decoded and the result thrown away. On
    gigapixel tail outputs that serialized minutes of vips decode per preflight
    satisfaction check on the StateActor (2026-08-06). The stamp must be
    returned without ever touching pixel data."""
    import numpy as np
    import file_io.generic as g
    from file_io.tiff_file_io import write_tiff_image

    p = tmp_path / "stamped.tiff"
    write_tiff_image(p, np.zeros((8, 8), np.float32), {"CONTHASH": "cafe1234"})

    def boom(f):
        raise AssertionError(f"eager pixel decode of stamped file {f}")
    monkeypatch.setattr(g, "hash_of_image_data", boom)

    assert g.get_content_hash(p) == "cafe1234"


# ------------------------------------------------- CALLHASH / CONTHASH split
def test_identity_prefers_a_declared_content_hash_then_the_call_hash(tmp_path):
    """Two header fields, two jobs. CALLHASH is the framework's
    write-completion witness and the DEFAULT identity; CONTHASH is optional and
    means "my producer declared an identity of its own". A file with neither
    falls back to hashing its pixels — the expensive branch, which is where raw
    frames legitimately land."""
    import numpy as np
    import astropy.io.fits as pyfits
    from file_io.generic import (
        get_content_hash, hash_of_image_data, read_call_hash,
        read_declared_content_hash,
    )
    from file_io.types import FitsFilePath

    def write(name, header):
        p = FitsFilePath(tmp_path / name)
        hdu = pyfits.PrimaryHDU(np.arange(16, dtype=np.float32).reshape(4, 4))
        for k, v in header.items():
            hdu.header[k] = v
        hdu.writeto(p, overwrite=True)
        return p

    plain = write("plain.fits", {"CALLHASH": "aaaa"})
    assert get_content_hash(plain) == "aaaa"                 # the call hash is the identity
    assert read_call_hash(plain) == "aaaa"
    assert read_declared_content_hash(plain) is None         # nothing declared

    declared = write("declared.fits", {"CALLHASH": "bbbb", "CONTHASH": "cccc"})
    assert get_content_hash(declared) == "cccc"              # the declaration wins
    assert read_call_hash(declared) == "bbbb"                # the witness is unchanged
    assert read_declared_content_hash(declared) == "cccc"

    bare = write("bare.fits", {})
    assert get_content_hash(bare) == hash_of_image_data(bare)
    assert read_call_hash(bare) is None


def test_an_all_digit_hash_survives_the_xisf_literal_eval(tmp_path):
    """The XISF reader runs `ast.literal_eval` on keyword values, so an
    all-digit hash comes back an int — and `4526235228981703 == '...'` is
    False. That made one master never-satisfied for a week. Both fields are
    read through `str()`; this pins it for the new one too."""
    import numpy as np
    from xisf import XISF
    from file_io.generic import get_content_hash, read_call_hash
    from file_io.types import XisfFilePath

    p = XisfFilePath(tmp_path / "digits.xisf")
    XISF.write(str(p), np.zeros((4, 4, 1), np.float32), creator_app="test",
               image_metadata={"FITSKeywords": {
                   "CALLHASH": [{"value": "4526235228981703", "comment": ""}],
                   "CONTHASH": [{"value": "1234567890123456", "comment": ""}]}, 
                   "XISFProperties": {}})
    assert read_call_hash(p) == "4526235228981703"
    assert get_content_hash(p) == "1234567890123456"
