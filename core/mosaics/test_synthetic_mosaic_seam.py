"""Apex test: synthetic source FITS → seamless mosaic, fully automatic.

Generates star-dense panels as ordinary input FITS — indistinguishable
from real camera frames except that we control vignetting, gradients,
sky background, and SNR — then runs the *real* AstroFlow pipeline
end-to-end (solve-field → target/mosaic identification → WBPP stack →
place_image → PhotometricMosaic merge) and checks the assembled mosaic
reads as a single flat photograph across the join.

This file writes NO pipeline code — only synthetic input generation
(`synthesize_calibration_suite` with `solvable_lights=True`). Everything
downstream is the production graph. The one non-obvious thing the apex
test must do is force `mock_pixinsight=False`: `test_context` defaults it
to True, which makes the merge copy the reference panel instead of
actually running PhotometricMosaic.

Tests, fastest-first:
  * `test_synthetic_csg_light_actually_plate_solves` — the keystone: a
    CSG-rendered star-dense synthetic light must solve with the real
    blind solve-field (the old NumPy-stars approach failed here).
  * `test_two_synthetic_panels_identified_as_mosaic` — two overlapping
    panels cluster into one mosaic in `context.mosaics`.
  * `test_synthetic_mosaic_is_seamless` — the full source-FITS → mosaic
    run via PixInsight Master; checks flatness across the join.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Where the mosaic + seam-profile artifacts land. Override with
# SEAM_TEST_IMAGE_DIR=/path.
_SEAM_IMAGE_DIR = Path(
    os.environ.get("SEAM_TEST_IMAGE_DIR", "/tmp/astroflow_seam_test")
)


# Geometry mirrors the known-good identification examples in
# test_identification.py: equatorial (Dec 0, so RA == on-sky degrees),
# axis-aligned (rotation 0). At 4096 px and image_scale 1"/px the FOV is
# 4096" = 1.1378°; an RA offset of 0.8·FOV leaves ~20% overlap — inside
# targets_overlap's 10–30% edge-thickness window.
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


# Apex field matrix: spans a MARS-covered galactic-plane field (→ MGC) and
# a high galactic-latitude field with no MARS coverage (→ ABE fallback),
# so the run exercises BOTH gradient-correction paths. Both at Dec 0
# (identification geometry validated equatorial; RA shifts galactic
# latitude). Each field's mosaic lands in /tmp/astroflow_seam_test/<field>/.
_APEX_FIELDS = {
    "galactic_plane_mgc": dict(ra_a=100.0, dec=0.0),    # b≈-2, MARS → MGC
    "high_lat_abe":       dict(ra_a=192.85, dec=0.0),   # b≈+63, no MARS → ABE
}


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
    the classic seam driver) so PhotometricMosaic has a real join to
    correct. Only synthetic input generation lives here — the pipeline
    solves, identifies, stacks, places, and PMM-merges on its own.
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


def _force_real_pixinsight() -> None:
    """test_context defaults mock_pixinsight=True (conftest), which makes
    WBPP/PhotometricMosaic no-ops that just copy an input.

    Principle: mock scales inversely with test level. Low-level/structural
    tests keep the mock so they're fast (graph wiring, identity, caching
    don't need real pixels). This apex test exists to judge *output
    quality* — whether the assembled mosaic is one seamless photograph —
    so it MUST process real pixels end to end. Flip the mock off."""
    from dataclasses import replace
    from context import context

    context.settings = replace(
        context.settings,
        development_settings=replace(
            context.settings.development_settings, mock_pixinsight=False,
        ),
    )


@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_diagnose_corner_glow(test_context, monkeypatch):
    """TEMP diagnostic: the apex mosaic shows a bright bottom-right corner
    glow even though the synthetic sky is uniform (no gradient). Disable
    ephemeral cleanup so every stage's output persists, then dump a
    thumbnail + a corner/center brightness ratio per stage. The stage
    where the ratio first jumps introduced the gradient (MGC in
    post_process_stack vs PhotometricMosaic's gradient extrapolation)."""
    from contextlib import nullcontext
    import numpy as np
    import pipelines.pixinsight_master as pim
    from compute_graph.compute_graph import build_compute_graph
    from context import context
    from dependency_graph import run_graph
    from file_io.generic import read_image
    from previews.operations import autostretch, array_to_png

    monkeypatch.setattr(pim, "ephemeral_intermediates", lambda: nullcontext())
    _force_real_pixinsight()
    add_synthetic_mosaic_images()
    run_graph(build_compute_graph())
    mid = next(iter(context.mosaics))
    context.shopping_cart = [("mosaic", mid, "PixInsight Master")]
    run_graph(build_compute_graph())

    out = _SEAM_IMAGE_DIR / "glow"
    out.mkdir(parents=True, exist_ok=True)

    def corner_ratio(a):
        """Median of the bottom-right corner vs the image center, valid
        pixels only. >1 means the BR corner is brighter (the glow)."""
        h, w = a.shape
        br = a[int(0.75*h):, int(0.75*w):]
        ce = a[int(0.4*h):int(0.6*h), int(0.4*w):int(0.6*w)]
        br = br[br > 0]; ce = ce[ce > 0]
        if br.size < 100 or ce.size < 100:
            return float("nan")
        return float(np.median(br) / max(np.median(ce), 1e-9))

    for op in ("stack_files", "post_process_stack", "place_image",
               "merge_images", "post_process_mosaic_color_channel"):
        d = context.output_dir / op
        files = sorted(d.glob("*.xisf")) + sorted(d.glob("*.tiff")) if d.exists() else []
        for i, f in enumerate(files):
            a = read_image(f).astype(np.float64)
            if a.ndim == 3:
                a = a[..., 0]
            print(f"[glow] {op}[{i}] {f.name}: shape={a.shape} "
                  f"corner/center={corner_ratio(a):.3f}")
            (out / f"{op}_{i}.png").write_bytes(
                array_to_png(autostretch(a.astype(np.float32)))
            )
    print(f"[glow] thumbnails in {out}")


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


def _seam_column_profile(mosaic) -> dict:
    """Per-column sky-background median across the assembled mosaic.

    The panels are offset in RA, so the join is roughly vertical — a
    column-wise sky profile is the natural seam probe: a flat mosaic is
    essentially one value everywhere, stars aside. For each column we
    take the median of valid, non-star pixels; a seam shows up as a step
    between the panels. Returns the profile plus its peak-to-peak spread
    relative to the sky noise."""
    import numpy as np

    valid = mosaic > 0
    med = float(np.median(mosaic[valid]))
    mad = float(np.median(np.abs(mosaic[valid] - med))) + 1e-12
    star = mosaic > med + 6.0 * 1.4826 * mad
    keep = valid & ~star

    W = mosaic.shape[1]
    prof = np.full(W, np.nan)
    for x in range(W):
        col = mosaic[:, x][keep[:, x]]
        if col.size > 50:
            prof[x] = np.median(col)

    good = np.where(np.isfinite(prof))[0]
    sky_pixels = mosaic[keep]
    sky_rms = float(np.std(sky_pixels[sky_pixels < np.percentile(sky_pixels, 75)]))
    # Peak-to-peak of the column sky profile over the valid span, ignoring
    # the outer 2% (edge-erosion ramps) at each end.
    inner = good[int(0.02 * good.size): max(1, int(0.98 * good.size))]
    pp = float(np.nanmax(prof[inner]) - np.nanmin(prof[inner])) if inner.size else float("nan")
    return {
        "profile": prof, "good_cols": good,
        "sky_median": med, "sky_rms": sky_rms,
        "seam_pp_adu": pp, "seam_over_rms": pp / max(sky_rms, 1e-12),
    }


def _join_step(seam: dict) -> dict:
    """Local seam discontinuity across the panel join — the field-
    independent 'is there a visible seam?' invariant. Smooth the column
    sky profile, then take the largest change over a ~50px window: a real
    join step is a sharp jump, whereas a per-panel smile or a smooth
    MARS-vs-synthetic gradient changes only gradually and is NOT a seam."""
    import numpy as np

    prof = seam["profile"]
    good = seam["good_cols"]
    if good.size < 200:
        return {"step_adu": float("nan"), "step_over_rms": float("nan")}
    # Measure the INTERIOR join, excluding the outer 2% at each end — the
    # same edge trim `_seam_column_profile` applies to the global p-p. The
    # extreme-edge columns are footprint-boundary / GC edge ramps (and on
    # synthetic MGC fields the synthetic↔MARS glow surfaces there); the
    # panel join lies in the interior, not at the mosaic edge.
    m = int(0.02 * good.size)
    inner = good[m: good.size - m] if good.size - 2 * m > 200 else good
    p = prof[inner]
    k = 25
    sm = np.convolve(p, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    W = 25
    if sm.size <= 2 * W:
        return {"step_adu": float("nan"), "step_over_rms": float("nan")}
    step = float(np.max(np.abs(sm[2 * W:] - sm[:-2 * W])))
    return {"step_adu": step,
            "step_over_rms": step / max(seam["sky_rms"], 1e-12)}


@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("field", list(_APEX_FIELDS))
def test_synthetic_mosaic_is_seamless(test_context, field):
    """Apex: source FITS → seamless mosaic, fully automatic, via the real
    PixInsight Master pipeline, across multiple sky regions.

    Two-pass (synthetic pixels make the stable hash run-dependent): run to
    identify the mosaic, drop it in the shopping cart, run again to build.

    Universal invariant asserted for EVERY field: the JOIN is seamless —
    no local step across the panel boundary (`_join_step`). Global flatness
    is reported but NOT asserted: on MARS/MGC fields, MGC subtracts the
    real-sky model from synthetic uniform data, leaving a smooth gradient
    that is a synthetic↔MARS mismatch, not a seam; only the data-driven ABE
    path is flat. Each field's mosaic + profile land in
    /tmp/astroflow_seam_test/<field>/ for visual comparison."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from compute_graph.compute_graph import build_compute_graph
    from context import context
    from dependency_graph import run_graph
    from file_io.generic import read_image
    from previews.operations import autostretch, array_to_png

    _force_real_pixinsight()
    add_synthetic_mosaic_images(**_APEX_FIELDS[field])

    # Pass 1 — solve + identify.
    run_graph(build_compute_graph())
    assert len(context.mosaics) == 1, (
        f"[{field}] identification produced {len(context.mosaics)} mosaics: "
        f"{list(context.mosaics)}"
    )
    mosaic_id = next(iter(context.mosaics))
    print(f"\n[apex:{field}] mosaic {mosaic_id}; building via PixInsight Master")

    # Pass 2 — build the mosaic through the real pipeline.
    context.shopping_cart = [("mosaic", mosaic_id, "PixInsight Master")]
    run_graph(build_compute_graph())

    final_dir = context.output_dir / "post_process_mosaic_color_channel"
    finals = sorted(final_dir.glob("*.xisf")) if final_dir.exists() else []
    assert finals, f"[{field}] no final mosaic under {final_dir}"
    mosaic = read_image(finals[-1]).astype(np.float64)
    if mosaic.ndim == 3:
        mosaic = mosaic[..., 0]

    out = _SEAM_IMAGE_DIR / field
    out.mkdir(parents=True, exist_ok=True)
    (out / "mosaic.png").write_bytes(
        array_to_png(autostretch(mosaic.astype(np.float32)))
    )

    seam = _seam_column_profile(mosaic)
    js = _join_step(seam)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(seam["good_cols"], seam["profile"][seam["good_cols"]],
            lw=0.7, color="0.2")
    ax.set_title(f"{field}: join step {js['step_over_rms']:.2f}× RMS, "
                 f"global p-p {seam['seam_over_rms']:.2f}× RMS")
    ax.set_xlabel("column (x) across mosaic")
    ax.set_ylabel("sky-background median (ADU)")
    fig.tight_layout()
    fig.savefig(out / "seam_profile.png", dpi=110)
    plt.close(fig)

    print(f"[apex:{field}] valid={float((mosaic>0).mean()):.2f} "
          f"sky_median={seam['sky_median']:.4g} sky_rms={seam['sky_rms']:.4g} "
          f"join_step={js['step_adu']:.4g} ({js['step_over_rms']:.2f}× RMS) "
          f"global_pp={seam['seam_pp_adu']:.4g} ({seam['seam_over_rms']:.2f}× RMS)")
    print(f"[apex:{field}] artifacts in {out}")

    assert np.isfinite(mosaic).all(), f"[{field}] mosaic has non-finite pixels."
    assert (mosaic > 0).mean() > 0.2, f"[{field}] mosaic is mostly empty."
    # Universal invariant: no visible seam at the join.
    assert js["step_over_rms"] < 2.0, (
        f"[{field}] visible seam at the join: {js['step_over_rms']:.2f}× sky "
        f"RMS (step {js['step_adu']:.4g} ADU)"
    )
