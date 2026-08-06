

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
