"""Contract tests for the type-keyed producer registry.

Producers are keyed by the runtime type of their first parameter. The
URL-side resolution uses a small string -> class map at the route layer.

Pure + fast (no Ray, no graph, no exhaustive fixture).
"""
import pytest


# --- per-type producer enumeration via MRO walk -----------------------

def test_producers_for_returns_decoration_order():
    """`producers_for(type)` returns each op whose first-parameter
    type accepts an instance of `type`. Walks MRO so an op annotated
    with a base class (`BaseStack`) surfaces for both subclasses
    (Stack, CalibrationStack)."""
    from products import ensure_producers_registered
    from products.registry import producers_for
    from stacks.composite_stack import Stack, CalibrationStack

    ensure_producers_registered()

    stack_ops = {p.__name__ for p in producers_for(Stack)}
    assert "stack_files" in stack_ops, (
        f"stack_files (annotated stack: BaseStack) must surface for "
        f"producers_for(Stack); got {stack_ops}"
    )
    assert "post_process_stack" in stack_ops

    cal_ops = {p.__name__ for p in producers_for(CalibrationStack)}
    # Same op surfaces for both subclasses via MRO walk.
    assert "stack_files" in cal_ops


# --- final=True on @node marks the user's deliverable --------------

def test_final_kwarg_propagates_to_wrapped_op():
    """`@node(final=True)` sets `_final` on the wrapped op so
    `dependency_graph.trace_paths.paths_for_cart_item(...,
    finals_only=True)` can filter to the user's preserve-on-archive
    set. Unmarked ops should lack `_final` (or have it False)."""
    from mosaics.post_process_channel import post_process_mosaic_color_channel
    from mosaics.final_storage import store_mosaic_color_channel_subframe
    from mosaics.merging.oss import merge_images_oss
    from stacks.post_processing.pixinsight import post_process_stack

    for op in (
        post_process_mosaic_color_channel,
        store_mosaic_color_channel_subframe,
        merge_images_oss,  # OSS deliverable: now display="Mosaic", final=True
    ):
        assert getattr(op, "_final", False) is True, (
            f"{op.__name__} should be marked final=True"
        )

    # post_process_stack is a per-Stack intermediate from the mosaic
    # user's perspective — should NOT be marked final.
    assert not getattr(post_process_stack, "_final", False), (
        "post_process_stack must not be final (intermediate from "
        "the mosaic user's perspective)"
    )


def test_final_requires_display():
    """`final=True` is a deliverable, always user-facing — so it requires an
    explicit `display=`. A final op without display raises at decoration."""
    from decorators import node
    from file_io.types import WithSidecar, XisfFilePath

    with pytest.raises(ValueError, match="final=True requires"):
        @node(final=True)
        def _bad(stack) -> WithSidecar[XisfFilePath]:
            ...

    # final + display is fine.
    @node(final=True, display="Master")
    def _ok(stack) -> WithSidecar[XisfFilePath]:
        ...
    assert _ok._final is True and _ok._display == "Master"


# --- image_type stamping (IMAGETYP) on producer outputs --------------

def _write_fits(path, header=None):
    import astropy.io.fits as pyfits
    import numpy as np
    hdu = pyfits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    for k, v in (header or {}).items():
        hdu.header[k] = v
    hdu.writeto(path, overwrite=True)


def test_stamp_header_adds_imagetyp_and_content_hash(tmp_path):
    """`_stamp_header` writes CONTHASH=call_hash and, by default,
    IMAGETYP=light into a produced file that declares no type — so
    `is_light_frame` is uniformly true for produced images."""
    from products.registry import _stamp_header
    from file_io.generic import read_header
    from file_io.types import FitsFilePath

    p = FitsFilePath(tmp_path / "out.fits")
    _write_fits(p)
    _stamp_header(p, "deadbeef", "light")

    header = read_header(p)
    assert header.get("CONTHASH") == "deadbeef"
    assert str(header.get("IMAGETYP")).lower() == "light"


def test_stamp_header_does_not_clobber_existing_imagetyp(tmp_path):
    """The shared stacker writes its own IMAGETYP ("Master Dark" etc.) in
    the body; `_stamp_header` must preserve it — only the content hash is
    added. This is what keeps the calibration carve-out working without
    touching stack_files."""
    from products.registry import _stamp_header
    from file_io.generic import read_header
    from file_io.types import FitsFilePath

    p = FitsFilePath(tmp_path / "master_dark.fits")
    _write_fits(p, {"IMAGETYP": "Master Dark"})
    _stamp_header(p, "cafe", "light")

    header = read_header(p)
    assert header.get("CONTHASH") == "cafe"
    # read_fits_header lowercases header values; the point is the body's
    # "Master Dark" survived (not overwritten by the default "light").
    assert str(header.get("IMAGETYP")).lower() == "master dark"  # not clobbered


def test_stamp_header_image_type_none_skips_imagetyp(tmp_path):
    """`image_type=None` disables the IMAGETYP stamp entirely; only the
    content hash is written."""
    from products.registry import _stamp_header
    from file_io.generic import read_header
    from file_io.types import FitsFilePath

    p = FitsFilePath(tmp_path / "untyped.fits")
    _write_fits(p)
    _stamp_header(p, "f00d", None)

    header = read_header(p)
    assert header.get("CONTHASH") == "f00d"
    assert header.get("IMAGETYP") is None
