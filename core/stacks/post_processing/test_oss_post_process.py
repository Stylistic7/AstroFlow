"""Statistics regression for OSS stack post-processing — incl. REAL GraXpert.

The OSS "mosaic renders dead white" bug: GraXpert already normalizes its output
to [0,1], but `post_process_stack_oss` divided by 65535 a SECOND time, collapsing
every pixel into the sky pedestal — a flat master with no signal tail. The unit
suite never caught it because every other OSS test runs `mock_pixinsight=True`,
which SKIPS GraXpert and reads the raw-ADU stack directly (where the /65535 is
correct). These tests exercise both paths on a real testdata light and assert the
post-processed master keeps a healthy linear histogram: sky parked at the
pedestal, with the full signal tail (faint nebulosity → saturated stars) intact.

The real-GraXpert test runs on the FULL frame as-is (no crop — a crop would strip
the very background gradient GraXpert exists to model). It's ~9 s on a 26 MP frame.
"""
import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from graxpert_cli import GraXpertNotFoundError, graxpert_base_command
from stacks.post_processing import oss as oss_module
from stacks.post_processing.background_pedestal import _SKY_QUANTILE
from stacks.post_processing.oss import corrected_normalized_pixels

# A real light with a genuine light-pollution gradient (M42, 26 MP) — the
# smallest full testdata light, so the real GraXpert run stays well under the
# 120 s per-test timeout.
_REAL_FRAME = Path(
    "testdata/input/Light_M42_180.0s_Bin1_2600MM_20231130-233344_0021.fit"
)
_PEDESTAL = 0.005


def _graxpert_available() -> bool:
    try:
        graxpert_base_command()
        return True
    except GraXpertNotFoundError:
        return False


def _sky_median(data: np.ndarray) -> float:
    """The same lower-quantile sky estimate the pedestal itself targets."""
    cutoff = np.quantile(data, _SKY_QUANTILE)
    return float(np.median(data[data <= cutoff]))


def _assert_healthy_linear_master(data: np.ndarray) -> None:
    """A correct OSS master: finite, positive, sky parked at the pedestal, and a
    long signal tail above it — i.e. NOT crushed flat (the white-mosaic bug)."""
    assert np.isfinite(data).all(), "post-processed master has non-finite pixels"
    assert data.min() >= 0.0, "negative pixels after pedestal"

    sky = _sky_median(data)
    assert abs(sky - _PEDESTAL) < 5e-4, f"sky {sky} not parked at pedestal {_PEDESTAL}"

    p50, p999, mx = (float(x) for x in np.percentile(data, [50, 99.9, 100]))
    # The double-normalize bug crushed the whole image into [pedestal, pedestal+1e-5]:
    # max would be ~0.005. A healthy linear master keeps highlights near full scale
    # and a broad tail rising out of the sky floor.
    assert mx > 0.5, f"highlights crushed (max={mx}) — the double-normalize regression"
    assert p999 - sky > 0.05, f"no signal tail above sky (p99.9={p999}, sky={sky})"
    # ...and it stays sky-DOMINATED, as a real deep image is: the median sits just
    # above the sky floor, far below the highlights.
    assert p50 < 0.1, f"bulk not sky-dominated (p50={p50})"


def test_post_process_stack_oss_delegates_plate_solving_to_decorator():
    """The op body must NOT call `solve_field` — plate-solving (and content-hash
    stamping) is owned by the `@node` decorator for EVERY image producer
    (`plate_solve="solve"` by default, run via `compute_image_metadata`).

    A manual solve in the body double-solves and, crucially, runs BEFORE the
    decorator stamps CONTHASH. That ordering let a master's on-disk
    content-hash fall back to a pixel hash (when the pre-stamp file state broke
    `update_header`), so downstream call_hashes diverged from the trace and the
    OSS color composites were swept as orphans and vanished after a run. Keep
    the body a pure writer; this guards the regression at the source level so a
    reintroduced solve fails fast (no external astrometry / Ray needed)."""
    tree = ast.parse(inspect.getsource(oss_module))
    body = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "post_process_stack_oss"
    )
    solves = [
        n for n in ast.walk(body)
        if isinstance(n, ast.Attribute) and n.attr == "solve_field"
    ]
    assert not solves, (
        "post_process_stack_oss calls solve_field in its body; plate-solving "
        "must be delegated to the @node decorator (plate_solve='solve' default)."
    )


@pytest.mark.skipif(not _REAL_FRAME.exists(), reason="testdata frame missing")
def test_mock_path_keeps_signal_tail():
    """Fast path (no GraXpert): raw 16-bit ADU /65535 + pedestal. Always runs."""
    data = corrected_normalized_pixels(str(_REAL_FRAME), True, _PEDESTAL)
    _assert_healthy_linear_master(data)


# `exhaustive` (opt-in): runs the real external GraXpert (~9 s solo, but it
# contends for CPU/GPU and can blow the 120 s timeout inside the parallel `-n`
# suite). Excluded from the default fast suite; run via `pytest -m exhaustive`.
# The mock-path test above still guards the statistics on every default run.
@pytest.mark.exhaustive
@pytest.mark.timeout(600)
@pytest.mark.skipif(not _REAL_FRAME.exists(), reason="testdata frame missing")
@pytest.mark.skipif(not _graxpert_available(), reason="GraXpert not installed")
def test_real_graxpert_keeps_signal_tail():
    """Production path: REAL GraXpert background-extraction + pedestal on a full
    real frame. The regression guard for the white-mosaic double-normalize —
    GraXpert emits [0,1], so the master must NOT be divided by 65535 again."""
    data = corrected_normalized_pixels(str(_REAL_FRAME), False, _PEDESTAL)
    _assert_healthy_linear_master(data)
