"""`analyze_master` measures visibility on a master and stores it in the
sidecar; more hours never reveal less, and noise is not mistaken for
structure. Runs as a script too."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from metadata import visibility as V
from metadata.master_analysis import FACTORS


def _field(rng, sigma, faint=3e-4, h=600, w=800):
    """Flat sky, a faint broad blob and a brighter knot, plus noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    blob = faint * np.exp(-(((xx - 500) / 120) ** 2 + ((yy - 300) / 90) ** 2))
    knot = 10 * faint * np.exp(-(((xx - 250) / 15) ** 2 + ((yy - 200) / 15) ** 2))
    return (0.005 + blob + knot + rng.normal(0, sigma, (h, w))).astype(np.float32)


def test_more_hours_reveal_more_and_noise_is_not_structure():
    rng = np.random.default_rng(1)
    deep, shallow = _field(rng, 2e-4), _field(rng, 4e-4)
    dv, dm = V.snr_maps(deep)
    sv, sm = V.snr_maps(shallow)
    fractions = [V.visible_fraction(dv, dm, f) for f in FACTORS]
    assert all(b >= a for a, b in zip(fractions, fractions[1:]))
    assert V.visible_fraction(sv, sm, 1.0) < V.visible_fraction(dv, dm, 1.0)
    # quality levels nest: what is clean is also usable and detected, and
    # deeper data lifts every level
    levels = V.quality_fractions(dv, dm, 1.0)
    assert levels == sorted(levels, reverse=True)
    assert all(b >= a for a, b in zip(levels, V.quality_fractions(dv, dm, 4.0)))
    empty = (0.005 + rng.normal(0, 2e-4, (600, 800))).astype(np.float32)
    ev, em = V.snr_maps(empty)
    assert V.visible_fraction(ev, em, 1.0) < 0.02


def test_analyze_master_writes_the_sidecar_section(tmp_path):
    from file_io.sidecar import get_section
    from file_io.tiff_file_io import write_tiff_image
    from metadata.master_analysis import SECTION, analyze_master
    path = tmp_path / "m.tiff"
    write_tiff_image(path, _field(np.random.default_rng(2), 2e-4), {"FILTER": "O", "STBLHASH": "abc"})

    class _M:
        filename, stable_hash, header, master_integration_s = path, "abc", {"FILTER": "O"}, 36000.0
    out = inspect.unwrap(analyze_master._function)(_M())
    assert get_section(path, SECTION) == out
    assert out["stack"] == "abc" and out["filter"] == "O"
    curves = out["visibility"]["fraction"]
    assert out["visibility"]["thresholds"] == list(V.THRESHOLDS)
    assert len(curves) == len(V.THRESHOLDS) and all(len(c) == len(FACTORS) for c in curves)
    assert all(b >= a for c in curves for a, b in zip(c, c[1:]))    # more hours never reveal less


if __name__ == "__main__":
    import tempfile
    test_more_hours_reveal_more_and_noise_is_not_structure(); print("ok more hours / noise")
    test_analyze_master_writes_the_sidecar_section(Path(tempfile.mkdtemp())); print("ok sidecar section")
