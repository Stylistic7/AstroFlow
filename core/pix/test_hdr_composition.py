"""Tests for the n>1 HDR merge path: StarAlignment registration + HDRComposition.

Regression context (all verified live against PixInsight 1.9.4): the n>1 path in
`targets/merging/pixinsight.py` was fully broken on three counts —
  1. HDRComposition was handed VIEW IDS, but its `images` parameter takes FILE
     PATHS (it opens the files itself); view ids made `executeGlobal()` return
     false with `*** Error: FileFormat: Empty format name…`, leaving no `HDR`
     view for the downstream save ("Output for view 'HDR' not found").
  2. It carried phantom params (`combinationMethod`/`HDRC_MaximumValue`,
     `outputHints`) that don't exist on the process.
  3. StarAlignment names registered targets `<id>_registered`, but the wrapper
     assumed `<id>_aligned`, so the aligned target views never existed.

The fast tests below lock the generated PJSR; the slow test exercises the whole
path on REAL PixInsight, synthesizing an HDR exposure pair from a SINGLE real
subframe (a 'long' whose bright cores are boosted to saturation + a 'short' that
recovers them) — a real star field for StarAlignment and a clean exposure ratio
for HDRComposition's linear fit, with no multi-exposure testdata needed.
"""
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.testdata  # reads core/testdata

from file_io.generic import read_image
from file_io.xisf_file_io import write_xisf_image
from pix import Image
from pix.tasks import HDRComposition, StarAlignment

_TEST_FRAME = Path(
    "testdata/input/Light_NGC7635_300.0s_Bin1_6200MM_H_20231004-213019_0004.fit"
)


def _lazy_images(tmp_path: Path, n: int):
    """`n` openable (existing) placeholder files → lazy Images. The bytes are
    never read: these tests inspect only the generated PJSR, not pixels."""
    files = []
    for i in range(n):
        p = tmp_path / f"f{i}.xisf"
        p.write_bytes(b"placeholder")
        files.append(str(p))
    return Image.open_many(files)


def test_hdr_composition_composes_files_not_views(tmp_path):
    """HDRComposition must dump each input view to a temp file and pass FILE
    PATHS (it opens files, not views), guard executeGlobal, and carry no
    phantom params."""
    images = _lazy_images(tmp_path, 2)
    composed = HDRComposition(images)
    js = composed.recipe.steps[-1].javascript

    # Each input view is written to a temp file first...
    assert js.count(".window.saveAs(") == 2
    assert "_hdrc_input_0.xisf" in js and "_hdrc_input_1.xisf" in js

    # ...and the images parameter references those FILE PATHS, never view ids.
    images_line = next(l for l in js.splitlines() if l.lstrip().startswith("P.images ="))
    assert "_hdrc_input_0.xisf" in images_line and "_hdrc_input_1.xisf" in images_line
    for img in images:
        assert img.view_id not in images_line

    # executeGlobal is checked (false -> loud throw, not a silent missing view).
    assert "if (!P.executeGlobal())" in js
    # output is auto-named "HDR"; default call requests no rename.
    assert composed.view_id == "HDR"

    # Phantom params must be gone (verified absent from the real process).
    for ghost in ("combinationMethod", "HDRC_MaximumValue", "outputHints"):
        assert ghost not in js


def test_star_alignment_renames_registered_to_aligned(tmp_path):
    """StarAlignment names registered targets '<id>_registered'; the wrapper
    must normalize every output (reference + targets) to '_aligned'."""
    images = _lazy_images(tmp_path, 3)
    aligned = StarAlignment(images)
    js = images[0].recipe.steps[-1].javascript

    ref, t1, t2 = (i.view_id for i in images)
    assert f'View.viewById("{ref}").id = "{ref}_aligned";' in js
    assert f'View.viewById("{t1}_registered").id = "{t1}_aligned";' in js
    assert f'View.viewById("{t2}_registered").id = "{t2}_aligned";' in js
    assert [a.view_id for a in aligned] == [
        f"{ref}_aligned", f"{t1}_aligned", f"{t2}_aligned",
    ]


@pytest.mark.slow
def test_hdr_compose_real_subframe_pair(tmp_path):
    """Full path on REAL PixInsight: an HDR exposure pair synthesized from one
    real subframe -> StarAlignment -> HDRComposition -> a valid HDR master."""
    assert _TEST_FRAME.exists(), f"Test frame not found: {_TEST_FRAME}"
    data = read_image(_TEST_FRAME).astype(np.float32)
    if data.ndim == 3:
        data = data[:, :, 0]
    # Normalize by a high percentile (not max, which is a single-pixel outlier)
    # so the 'long' has a substantial saturated region (~0.3% of pixels) for
    # HDRComposition to detect and replace — under-saturation gives a degenerate
    # linear fit. 'short' is the same scene at 1/ratio, recovering those cores.
    ceil = max(float(np.percentile(data, 99.7)), 1.0)
    ratio = 6.0
    long = np.clip(data / ceil, 0.0, 1.0).astype(np.float32)
    short = np.clip(data / ceil / ratio, 0.0, 1.0).astype(np.float32)

    lf, sf, out = tmp_path / "long.xisf", tmp_path / "short.xisf", tmp_path / "hdr.xisf"
    write_xisf_image(lf, long[:, :, None], {"EXPTIME": 300})
    write_xisf_image(sf, short[:, :, None], {"EXPTIME": int(300 / ratio)})

    images = Image.open_many([str(lf), str(sf)], filters=["Ha", "Ha"])  # longest-first
    aligned = StarAlignment(images)
    composed = HDRComposition(aligned)
    assert composed.view_id == "HDR"
    composed.save_to(out, mock=False)   # real PixInsight

    assert out.exists() and out.stat().st_size > 0
    log = composed.recipe.log
    assert log and "HDR composition of 2 images" in log, f"HDRComposition did not run.\n{log}"
    assert "Performing HDR composition: done" in log, f"HDR compose failed.\n{log}"
    assert "bad linear fit" not in log

    result = read_image(out)
    assert result.shape[:2] == long.shape
    assert np.all(np.isfinite(result))
