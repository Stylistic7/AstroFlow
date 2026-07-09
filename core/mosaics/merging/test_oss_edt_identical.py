"""The OSS merge's distance-transform weight switched from scipy's
single-threaded `distance_transform_edt` to the multithreaded `edt`
package. Both are exact Euclidean transforms; these tests pin that the
swap leaves `_blend_many`'s output numerically identical (the user's
hard constraint), through the real function — only the EDT impl differs.

Fixture-free (manual tempdir + monkeypatch) so they can run directly,
Ray-free, without starting Ray against the live cluster.
"""
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import mosaics.merging.oss as oss
from mosaics.placement.panel_io import write_placed_panel


def _write_panel(fp: Path, img: np.ndarray) -> None:
    """Persist a full-canvas panel as bbox-on-disk (place_image's format)."""
    r0, r1, c0, c1 = oss._panel_bbox(img > 0)
    write_placed_panel(
        fp, img[r0:r1, c0:c1].astype(np.float32),
        offset_xy=(c0, r0), canvas_size=(img.shape[1], img.shape[0]),
    )


class _ScipyEDT:
    """Drop-in for the `edt` module exposing the old scipy transform with
    the same call signature `_blend_many` uses."""
    @staticmethod
    def edt(arr, black_border=False, parallel=0):
        return distance_transform_edt(arr.astype(bool))


def _write_synth_panels(d: Path) -> list[Path]:
    """Three overlapping gradient panels on a 100x140 canvas, leaving a
    zero gap (no-coverage region). Gradients make the pairwise overlap
    means differ so the flux solve is exercised."""
    h, w = 100, 140
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 1.0 + 0.15 * (xx / w) + 0.1 * (yy / h)  # smooth, all > 0

    def panel(y0, y1, x0, x1, level):
        img = np.zeros((h, w), np.float32)
        img[y0:y1, x0:x1] = (grad[y0:y1, x0:x1] * level)
        return img

    panels = [
        panel(10, 70, 10, 80, 1.00),   # A
        panel(40, 95, 50, 130, 0.92),  # B (overlaps A)
        panel(10, 50, 90, 135, 1.07),  # C (overlaps B)
    ]
    paths = []
    for i, p in enumerate(panels):
        fp = d / f"panel_{i}.tiff"
        _write_panel(fp, p)
        paths.append(fp)
    return paths


def test_blend_many_edt_matches_scipy():
    """Same `_blend_many`, edt vs scipy EDT → numerically identical."""
    with tempfile.TemporaryDirectory() as td:
        paths = _write_synth_panels(Path(td))
        new = oss._blend_many(paths)  # edt (multithreaded)
        orig = oss.edt
        try:
            oss.edt = _ScipyEDT
            old = oss._blend_many(paths)  # scipy reference
        finally:
            oss.edt = orig
        assert new.shape == old.shape
        assert np.allclose(old, new, rtol=1e-5, atol=1e-3), (
            f"max abs diff {np.abs(old - new).max():.3e}"
        )


def test_no_coverage_region_is_zero():
    """A pixel covered by no panel (the gap) blends to 0 (denom→1)."""
    with tempfile.TemporaryDirectory() as td:
        paths = _write_synth_panels(Path(td))
        out = oss._blend_many(paths)
        # bottom-left corner is outside every panel rect
        assert out[95:100, 0:10].max() == 0.0


def test_single_panel_returns_input():
    """N=1 short-circuits before any EDT and returns the panel as-is."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        img = np.zeros((40, 50), np.float32)
        img[5:35, 5:45] = 1.23
        fp = d / "solo.tiff"
        _write_panel(fp, img)
        out = oss._blend_many([fp])
        assert np.array_equal(out, img)


def test_edt_matches_scipy_on_realistic_masks():
    """The only changed primitive: edt == scipy on panel-like masks
    (filled regions with zero borders; never all-foreground)."""
    rng = np.random.default_rng(0)
    masks = {
        "rect-edge": np.zeros((120, 160), bool),
        "blob-holes": np.zeros((100, 100), bool),
        "random": rng.random((80, 80)) > 0.3,
    }
    masks["rect-edge"][5:115, 8:152] = True
    masks["blob-holes"][20:80, 20:80] = True
    masks["blob-holes"][40:50, 40:50] = False
    for name, m in masks.items():
        a = distance_transform_edt(m).astype(np.float32)
        b = oss.edt.edt(m.astype(np.uint8), black_border=False,
                        parallel=0).astype(np.float32)
        assert np.allclose(a, b, rtol=1e-5, atol=1e-3), name
