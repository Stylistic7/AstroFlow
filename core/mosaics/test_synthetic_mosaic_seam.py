"""Synthetic source FITS through the real front of the pipeline.

Star-dense panels are generated as ordinary input FITS — indistinguishable
from real camera frames except that we control vignetting, gradients, sky
background, and SNR — and handed to the production graph. This file writes
NO pipeline code, only synthetic input generation
(`synthesize_calibration_suite` with `solvable_lights=True`).

Tests, fastest-first:

  * `test_synthetic_csg_light_actually_plate_solves` — the keystone: a
    CSG-rendered star-dense synthetic light must solve with the real
    blind solve-field (the old NumPy-stars approach failed here).
  * `test_two_synthetic_panels_identified_as_mosaic` — two overlapping
    panels cluster into one mosaic in `context.mosaics`.

The seam itself is judged headless, on `merge_images_oss`, in
mosaics/merging/test_oss_seam.py.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


# Geometry mirrors the known-good identification examples in
# test_identification.py: equatorial (Dec 0, so RA == on-sky degrees),
# axis-aligned (rotation 0). At 4096 px and image_scale 1"/px the FOV is
# 4096" = 1.1378°; an RA offset of 0.8·FOV leaves ~20% overlap, well past
# targets_overlap's minimum area fraction.
#
# Field is at RA 192.85, Dec 0 — galactic latitude b ≈ +63°, deliberately
# OFF the galactic plane. The earlier default (RA 100, Dec 0) sits at
# b ≈ -2° in the Milky Way, where CSG renders a real, structured dense
# star field; that real unresolved-star background showed up as a corner
# "glow" in the assembled mosaic. A high-|b| field is sparse and
# structureless, so the seam/flatness checks measure the pipeline, not
# real sky structure.
_FOV_DEG = 4096 * 1.0 / 3600.0
_RA_OFFSET_DEG = 0.80 * _FOV_DEG  # ~20% panel overlap


def add_synthetic_mosaic_images(
    *, dec: float = 0.0, ra_a: float = 192.85,
    ra_b: float | None = None,
    width: int = 4096, height: int = 4096, n_light: int = 4,
    star_max_magnitude: float = 18.5, star_peak_adu: float = 2_000_000.0,
    sky_a_e_per_sec: float = 0.5, sky_b_e_per_sec: float = 1.5,
) -> None:
    """Lay down two overlapping star-dense panels as raw, real-looking
    input FITS in `context.input_dirs[0]`.

    Panel A also supplies the shared bias/dark/flat (one calibration
    session for the whole mosaic, as on a real rig); panel B contributes
    only its lights. The two panels differ in sky brightness (a 3× drift,
    the classic seam driver) so the merge has a real join to correct. Only
    synthetic input generation lives here — the pipeline solves, identifies,
    stacks, places and merges on its own.
    """
    from context import context
    from testing.calibration_physics import synthesize_calibration_suite

    if ra_b is None:
        ra_b = ra_a + _RA_OFFSET_DEG

    input_dir = Path(context.input_dirs[0])

    def _panel(ra, seed, prefix, sky, include_cals):
        tmp = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
        truth = synthesize_calibration_suite(
            output_dir=tmp, width=width, height=height,
            ra=ra, dec=dec, rng_seed=seed, n_light=n_light,
            sky_e_per_sec=sky,
            star_max_magnitude=star_max_magnitude,
            star_peak_adu=star_peak_adu,
            # Equatorial + axis-aligned → RA-offset panels share a clean
            # vertical edge for mosaic identification's edge-overlap test.
            rotation_angle_deg=0.0,
            solvable_lights=True, prefix=prefix,
        )
        files = list(truth.light_files)
        if include_cals:
            files += truth.bias_files + truth.dark_files + truth.flat_files
        for f in files:
            shutil.copy(f, input_dir / Path(f).name)

    _panel(ra_a, 0xA101, "synthpane_a", sky_a_e_per_sec, include_cals=True)
    _panel(ra_b, 0xB101, "synthpane_b", sky_b_e_per_sec, include_cals=False)


@pytest.mark.slow
@pytest.mark.timeout(420)
def test_synthetic_csg_light_actually_plate_solves(tmp_path):
    """A star-dense synthetic light, written as a real-looking input
    frame (no DONTSOLV, no baked solution), must solve via the real
    blind solve-field — exactly as a real camera frame would. Until this
    holds, the source-FITS → mosaic apex pipeline can't run on synthetic
    input."""
    import plate_solving
    from testing.calibration_physics import synthesize_calibration_suite

    truth = synthesize_calibration_suite(
        output_dir=tmp_path / "cal",
        width=4096, height=4096,
        n_bias=1, n_dark=1, n_flat=1, n_light=1,
        ra=100.0, dec=45.0,
        star_max_magnitude=18.5,
        # Saturate bright stars so the faint field rises above noise (a
        # realistic detectable star count for solve-field to lock onto).
        star_peak_adu=2_000_000.0,
        solvable_lights=True,
        prefix="solvecheck",
    )

    light = Path(truth.light_files[0])
    ps = plate_solving.solve_field(light, debug=True)

    assert ps is not None, (
        f"solve-field could not solve the synthetic CSG light {light}. "
        f"The star field is too sparse/faint for a blind solve — raise "
        f"star_max_magnitude or star_peak_adu."
    )
    print(f"\n[solve] CSG synthetic light solved: "
          f"center={ps.center} scale={ps.image_scale}")


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_two_synthetic_panels_identified_as_mosaic(test_context):
    """Through the REAL solve path: two overlapping star-dense panels must
    cluster into a single mosaic in `context.mosaics` with a stable hash
    (the handle the rest of the pipeline keys off)."""
    from compute_graph.compute_graph import build_compute_graph
    from context import context
    from dependency_graph import run_graph

    add_synthetic_mosaic_images()
    # store_targets / store_mosaics populate context.{targets,mosaics} only
    # when they execute, so the graph must run. An empty shopping cart
    # still solves + identifies (foundational) without the heavy
    # WBPP/place/merge work.
    run_graph(build_compute_graph())

    print(f"\n[identify] targets={len(context.targets)} "
          f"mosaics={len(context.mosaics)}")
    assert len(context.mosaics) == 1, (
        f"Expected one mosaic from two overlapping panels, got "
        f"{len(context.mosaics)}: {list(context.mosaics)}"
    )


