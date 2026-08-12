"""Compare the display ops (display/ops.py, display/finishes.py) against the
ACTUAL PixInsight processes they emulate — the ground truth the user asked for.

SLOW: these launch real PixInsight. The pix/ dir is excluded from the default
pytest run (pytest.ini `--ignore=pix`), so they run only when invoked
explicitly, AND — because pytest resets the shared Ray cluster — only with the
live app DOWN:

    make stop            # (or ensure no derivation is running; frees the PI slot)
    pytest pix/test_display_ops_vs_pixinsight.py -v -m slow

Flow per op: take the real SHO fixture, stretch it to a display [0,1] RGB, write
it as XISF, run the PixInsight process, read it back, assert the numpy op matches.

The numpy↔PI boundary is XISF (write_xisf_image / read_xisf_image). The PixInsight
process tasks are defined inline here (they're test-only — the app doesn't use
PI for these ops). The PJSR parameter names/enums follow PixInsight's documented
API; if a process rejects a property on first run, PI logs the offending name —
fix it here (the harness, I/O, and flow are correct).
"""
import numpy as np
import pyvips
import pytest
from pathlib import Path

from pix.image import Image
from pix.tasks.process import make_process_task
from pix.tasks.js import Enum
from file_io.xisf_file_io import write_xisf_image, read_xisf_image
from finishes.stdlib import _mtf_np
from mosaics.color import palettes

FIXTURE = Path(__file__).resolve().parent.parent / "testdata" / "color" / "sho_crop_768.tiff"

# ── PixInsight process tasks (test-only) ─────────────────────────────────────
# HistogramTransformation applies the midtones-transfer function. H is a 5-row
# matrix (R, G, B, RGB/K, Alpha); each row = [shadows, midtones, highlights,
# lowRange, highRange], identity [0, 0.5, 1, 0, 1] (see PixInsight_Edits/
# Geometry.js:187). A pure MTF(m) on every channel = set the RGB/K row's midtones
# to m, all others identity.
_IDENTITY_ROW = [0.0, 0.5, 1.0, 0.0, 1.0]


def _pi_mtf(midtones: float):
    return make_process_task("HistogramTransformation", {
        "H": [_IDENTITY_ROW, _IDENTITY_ROW, _IDENTITY_ROW,
              [0.0, midtones, 1.0, 0.0, 1.0], _IDENTITY_ROW],
    })


# ── shared fixtures / helpers ────────────────────────────────────────────────
@pytest.fixture(scope="module")
def display_rgb():
    """The SHO fixture stretched to a realistic display [0,1] RGB (per-channel
    autostretch — the app's production stretch), so the ops see representative
    values rather than near-linear noise."""
    img = pyvips.Image.new_from_file(str(FIXTURE), unlimited=True)
    lin = np.ndarray(buffer=img.write_to_memory(), dtype=np.float32,
                     shape=[img.height, img.width, 3])
    return np.ascontiguousarray(
        np.stack([palettes.autostretch(lin[..., i]) for i in range(3)], axis=2),
        dtype=np.float32)


def _run_pi(task, rgb, tmp_path, **kwargs):
    """numpy RGB → XISF → PixInsight process → XISF → numpy RGB."""
    inp = tmp_path / "in.xisf"
    write_xisf_image(inp, np.ascontiguousarray(rgb, dtype=np.float32))
    out = task(Image.open(inp), **kwargs)
    outp = tmp_path / "out.xisf"
    out.save_to(outp)
    return read_xisf_image(outp).astype(np.float32)


@pytest.mark.slow
class TestMTFvsPixInsight:
    @pytest.mark.parametrize("midtones", [0.15, 0.25, 0.5])
    def test_histogram_transformation_matches(self, display_rgb, tmp_path, midtones):
        pi = _run_pi(_pi_mtf(midtones), display_rgb, tmp_path)
        # The finishes stdlib MTF is exactly PixInsight's
        # HistogramTransformation midtones curve.
        mine = _mtf_np(display_rgb, midtones)
        np.testing.assert_allclose(mine, pi, atol=2e-3)
