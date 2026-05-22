"""Regression test for the PI Downscaled pipeline's defining invariant:

The `stack_files_downscaled` op feeds INPUT frames into WBPP at 1/8
resolution (block-mean), not full resolution. The whole point of
the pipeline — WBPP / plate-solving / registration running ~64x
cheaper — depends on this being correct, so it's worth a focused
test that doesn't have to run WBPP to verify.

Tests:

* `_copy_frames_with_plate_solution(..., downscale_factor=8)` calls
  `downscale_fits` with K=8 for every input frame. The output FITS
  files are 8x smaller per side.

* `stack_files_downscaled` is registered for both StackId and
  CalibrationStackId — calibration frames are downscaled BEFORE
  WBPP too, so the resulting cal masters match the (downscaled)
  light frames they'll be calibrating.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from previews.types import DOWNSCALED_MASTER, DOWNSCALE_FACTORS


def test_downscaled_factor_is_8():
    """The downscaled pipeline uses a factor of 8 (1/64 the pixel
    count). If anyone tunes this constant they need to update both
    the test and the docs that quote the speedup ratio."""
    assert DOWNSCALE_FACTORS[DOWNSCALED_MASTER] == 8


def test_copy_frames_downscales_input_frames():
    """`_copy_frames_with_plate_solution(downscale_factor=8)` calls
    downscale_fits with K=8 for every frame. This is the load-bearing
    fact — WBPP sees 64x-smaller frames because of this call."""
    from stacks.stacking.pixinsight import _copy_frames_with_plate_solution

    # Build a minimal MetadataDict-shaped fake.
    class _FakeFrame:
        def __init__(self, filename):
            self.filename = filename
            self.plate_solution = None

    with tempfile.TemporaryDirectory() as tmp:
        src_filename = os.path.join(tmp, "frame_00.fits")
        # Touch the source so the existence check passes.
        Path(src_filename).touch()
        dest_dir = os.path.join(tmp, "wbpp_input")
        os.makedirs(dest_dir, exist_ok=True)

        metadata = {"frame_00": _FakeFrame(src_filename)}

        with patch(
            "stacks.downscale.downscale_fits"
        ) as mock_downscale:
            _copy_frames_with_plate_solution(metadata, dest_dir, downscale_factor=8)

        assert mock_downscale.call_count == 1, (
            "downscale_fits should be called once per input frame "
            "when downscale_factor > 1"
        )
        # Positional args: (src, dest); kwargs: K=8.
        args, kwargs = mock_downscale.call_args
        assert kwargs.get("K") == 8, (
            f"downscale_fits should be invoked with K=8, got K={kwargs.get('K')}"
        )


def test_copy_frames_does_not_downscale_at_factor_1():
    """Default (downscale_factor=1) MUST NOT touch downscale_fits —
    the full-res `stack_files` pipeline depends on the input frames
    being untouched copies."""
    from stacks.stacking.pixinsight import _copy_frames_with_plate_solution

    class _FakeFrame:
        def __init__(self, filename):
            self.filename = filename
            self.plate_solution = None

    with tempfile.TemporaryDirectory() as tmp:
        src_filename = os.path.join(tmp, "frame_00.fits")
        # Need a real FITS-shaped file for the non-downscale branch's
        # pyfits.open to succeed; stub with a minimal valid FITS.
        from astropy.io import fits as pyfits
        hdu = pyfits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.uint16))
        hdu.writeto(src_filename)
        dest_dir = os.path.join(tmp, "wbpp_input")
        os.makedirs(dest_dir, exist_ok=True)

        metadata = {"frame_00": _FakeFrame(src_filename)}

        with patch(
            "stacks.downscale.downscale_fits"
        ) as mock_downscale:
            _copy_frames_with_plate_solution(metadata, dest_dir, downscale_factor=1)

        assert mock_downscale.call_count == 0, (
            "downscale_fits should NOT be called when downscale_factor=1"
        )


def test_stack_files_downscaled_registered_for_both_identities():
    """Cal masters must also come from `stack_files_downscaled` (the
    PI Downscaled pipeline's `calibrate_downscaled` uses
    stack_files_downscaled as its stacker). The op is registered for
    BOTH StackId and CalibrationStackId so it can produce light AND
    cal masters."""
    from stacks.stacking.pixinsight import stack_files_downscaled
    from products.identity import StackId, CalibrationStackId

    keys = stack_files_downscaled.filename_fns.keys()
    identities = {k[0] for k in keys}
    assert StackId in identities, (
        f"stack_files_downscaled should be registered for StackId, "
        f"got keys: {list(keys)}"
    )
    assert CalibrationStackId in identities, (
        f"stack_files_downscaled should be registered for "
        f"CalibrationStackId so cal masters get downscaled too, "
        f"got keys: {list(keys)}"
    )


def test_calibrate_downscaled_uses_stack_files_downscaled():
    """`calibrate_downscaled` MUST use `stack_files_downscaled` as
    its stacker. If it accidentally uses the full-res `stack_files`,
    cal masters would be full-res and the light-stack downscale
    would try to symlink incompatible (full-res) cal masters."""
    import inspect
    from stacks.compute_graph import calibrate_downscaled

    src = inspect.getsource(calibrate_downscaled._function if hasattr(calibrate_downscaled, '_function') else calibrate_downscaled)
    assert "stack_files_downscaled" in src, (
        "calibrate_downscaled body should reference "
        "stack_files_downscaled (so cal stacks get downscaled too)"
    )
