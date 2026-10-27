"""4-panel narrowband mosaic lab: Cygnus-calibrated errors, real machinery.

A 2x2 mosaic (one closure loop — the solver's hard case) of ~1MP panels in
H/S/O, corrupted with the error model measured on Cygnus (gain spread to
1.79x worst-pair, per-panel sky constants ~1e-4, varying noise), then run
through the real pipeline pieces:

  candidate chain:  SPFC per panel -> gain-normalize -> the REAL
                    merge_images_oss @node (traced + run_graph) [asserted]
  current chain:    the real merge op alone (internal flux solve +
                    offset solve, no SPFC)                      [reported]
  baseline chain:   uncorrupted panels through the same op — the
                    chroma metric's scene floor
  control:          last-writer overlay, no correction          [must fail]

Judged by the shared seam metric per channel (both join orientations) and
the stretched-chroma metric — cross-channel background disagreement AFTER
the canonical display autostretch, i.e. in the units the eye uses —
referenced to the baseline chain, since per-channel autostretch of
genuinely different nebulosity has a nonzero scene floor.

Deliberately absent this round: within-panel gradients and GraXpert
polish. Constants+gains are what today's solves claim to fix; gradients
(and the affine-tilt solve / polish experiments) are the next iteration
of this lab. Note polish's GraXpert 256^2 internal model is relatively
FINER on a small lab mosaic than on gigapixel production — this lab can
prove correctness but not polish's resolution limit.

Requires PixInsight + local Gaia DR3/SP XPSD. NEVER run while the app is
up (pytest resets the shared Ray cluster).
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

import numpy as np
import pytest

_RA, _DEC = 100.0, 0.0
# Fixed panel size, canvas grows with the grid (single-variable scaling —
# see test_lab_full_pipeline).
_PANEL_H, _PANEL_W = 800, 1200
_SCALE_DEG = 2.0 / 3600.0
_GRID = (4, 4)
_OVERLAP = 0.25
_CANVAS_H = int(_PANEL_H * (_GRID[0] - (_GRID[0] - 1) * _OVERLAP))
_CANVAS_W = int(_PANEL_W * (_GRID[1] - (_GRID[1] - 1) * _OVERLAP))
_MAG14_TARGET = 0.0078  # production flux gauge — see E_AT_PEDESTAL_RANGE note                  # linear units, total flux of a G=14 star

# FITS-standard sky parity (CD1_1 < 0: RA grows leftward, det < 0) — the
# representation every real frame has. PI's legacy WCS path MIRRORS the
# unphysical +/+ diagonal (found 2026-09-01: same matrix, represented as
# CROTA 0 with positive CDELTs, projects y-flipped through PI while the
# equivalent CROTA-180/negative-CDELT form does not).
_CANVAS_WCS = {
    "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
    "CRVAL1": _RA, "CRVAL2": _DEC,
    "CRPIX1": _CANVAS_W / 2 + 0.5, "CRPIX2": _CANVAS_H / 2 + 0.5,
    "CD1_1": -_SCALE_DEG, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": _SCALE_DEG,
}

_READ_FLX_JS = """
let view = View.viewById("{view_id}");
let sf = view.propertyValue("PCL:SPFC:ScaleFactors");
let flx = null;
if (sf != null)
    flx = (typeof sf == "number") ? sf : (sf.at ? sf.at(0) : sf[0]);
if (flx == null || !isFinite(flx) || flx <= 0)
    throw new Error("mosaic lab: no PCL:SPFC:ScaleFactors after SPFC");
console.writeln(format("ASTROFLOW: mosaiclab SPFCSCALE %.8e", flx));
"""


def _star_scale(catalog: dict, filter_key: str) -> float:
    from testing.narrowband_stars import band_flux

    fluxes = np.array([
        band_flux(catalog["wavelength_nm"], spec, filter_key)
        for _, _, g, spec in catalog["stars"] if abs(g - 14.0) < 0.75
    ])
    assert fluxes.size >= 5
    return _MAG14_TARGET / float(np.median(fluxes))


def _spfc_scale(panel_fits: Path, filter_key: str, wcs: dict,
                out: Path) -> float:
    from pix.image import Image
    from pix.recipe import Step
    from pix.tasks import (
        SpectrophotometricFluxCalibration,
        WriteAstrometricSolution,
    )

    img = Image.open(panel_fits, filter=filter_key)
    img = WriteAstrometricSolution(img, wcs)
    img = SpectrophotometricFluxCalibration(img)
    img.recipe.add_step(Step(
        name="read SPFC scale",
        javascript=_READ_FLX_JS.format(view_id=img.view_id),
    ))
    img.save_to(out)
    found = re.findall(r"mosaiclab SPFCSCALE ([0-9.eE+-]+)", img.recipe.log)
    assert found, f"no SPFC scale for {panel_fits.name}"
    return float(found[-1])


_DATE = datetime.datetime(2026, 4, 18, 6, 0, 0)


def _lab_mosaic(tag: str):
    """Minimal real Mosaic, the sanctioned test construction (see
    mosaics/color/test_post_process.py) — distinct hashes per merge call
    so each variant/channel keys its own cache entry."""
    from frozendict import frozendict
    from spherical_geometry.polygon import SingleSphericalPolygon

    from attributes.astrometric_attributes import healpix_cells_from_poly
    from mosaics.types import Mosaic
    from units import ArcsecondsPerPixel, Degrees, Radians

    footprint = SingleSphericalPolygon.from_radec(
        [99.4, 100.6, 100.6, 99.4], [-0.4, -0.4, 0.4, 0.4],
        center=[_RA, _DEC])
    return Mosaic(
        name=f"lab-{tag}", content_hash=f"lab_content_{tag}",
        stable_hash=f"lab_{tag}", target_ids=("a",),
        panel_target_ids=("a",), oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(_RA), Degrees(_DEC)), stable_center=(Degrees(_RA), Degrees(_DEC)),
        image_scale=ArcsecondsPerPixel(2.0), rotation_angle=Radians(0.0),
        footprint=footprint, healpix_cells=healpix_cells_from_poly(footprint),
        projected_footprints=frozendict(),
    )


def _merge_flow(mosaic, filter, *metas):
    from mosaics.merging.oss import merge_images_oss

    return merge_images_oss(mosaic, filter, *metas)


def _merge(tmp: Path, tag: str, ch: str, panels: list, specs: list) -> np.ndarray:
    """The REAL pipeline merge: the merge_images_oss @node op, traced and
    executed through run_graph — not a reimplementation of its internals."""
    from dependency_graph import run_graph
    from dependency_graph.tracing import trace
    from file_io.generic import read_image
    from metadata.classes import ImageMetadata
    from mosaics.merging.oss import merge_images_oss
    from testing.synthetic_mosaic import write_placed_panel_file

    metas = []
    for i, (p, s) in enumerate(zip(panels, specs)):
        path = write_placed_panel_file(
            tmp / f"{tag}_{ch}_{i}.tiff", p, s, _CANVAS_H, _CANVAS_W)
        # Direct construction is the sanctioned harness: the op reads only
        # `.filename`/`.stable_hash`; `content_hash` keys the cache.
        metas.append(ImageMetadata(
            content_hash=f"{tag}_{ch}_{i}", stable_hash=f"{tag}_{ch}_{i}",
            filename=path, header={"FILTER": ch},
            width_pixels=s.width, height_pixels=s.height,
            channels=1, dtype=np.dtype("float32"),
        ))
    mosaic = _lab_mosaic(f"{tag}_{ch}")
    graph = trace(_merge_flow, mosaic, ch, *metas)
    run_graph(graph)
    out = Path(merge_images_oss.filename_fn(mosaic, ch, *metas))
    merged = read_image(out).astype(np.float64)
    if merged.ndim == 3:
        merged = merged[..., 0]
    return merged


def _naive_overlay(panels: list, specs: list) -> np.ndarray:
    canvas = np.zeros((_CANVAS_H, _CANVAS_W), np.float64)
    for p, s in zip(panels, specs):
        canvas[s.row0:s.row0 + s.height, s.col0:s.col0 + s.width] = p
    return canvas


def _join_steps(merged: np.ndarray) -> tuple:
    from testing.seam_metrics import join_step, seam_column_profile

    v = join_step(seam_column_profile(merged))
    h = join_step(seam_column_profile(np.ascontiguousarray(merged.T)))
    return v["step_over_rms"], h["step_over_rms"]


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_current_chain_chroma_sweep(tmp_path):
    """Seed sweep of the CURRENT (no-SPFC) chain: corrupted panels through
    the real merge op, chroma spread vs each seed's own zero-error
    baseline. Answers "does the production pipeline quilt, and how often?"
    with a distribution instead of one draw (the measured penalty ranged
    0.001-0.066 display units across the first two draws). Renders the
    worst draw to $ASTROFLOW_LAB_RENDERS for eyeballing. PixInsight-free."""
    import os

    from testing.canonical_render import hubble_render, save_png
    from testing.seam_metrics import stretched_chroma_spread
    from testing.synthetic_mosaic import (
        corrupt_panel,
        make_panel_specs,
        make_truth_skies,
        PanelSpec,
    )

    render_dir = Path(os.environ.get("ASTROFLOW_LAB_RENDERS",
                                     "/tmp/astroflow_lab"))
    render_dir.mkdir(parents=True, exist_ok=True)

    # One shared scene (stars matter less here — reuse a light star field
    # drawn synthetically to keep the sweep PI-free; chroma is measured on
    # star-rejected background regions anyway).
    scene_rng = np.random.default_rng(0xC0FFEE)
    fake_stars = {}
    for ch in ("H", "S", "O"):
        layer = np.zeros((_CANVAS_H, _CANVAS_W))
        ys = scene_rng.integers(8, _CANVAS_H - 8, 400)
        xs = scene_rng.integers(8, _CANVAS_W - 8, 400)
        layer[ys, xs] = scene_rng.uniform(0.01, 0.4, 400)
        fake_stars[ch] = layer
    skies = make_truth_skies(_CANVAS_H, _CANVAS_W, fake_stars, scene_rng)

    from testing.seam_metrics import truth_chroma_bias

    # Floor reference: one zero-error draw through the same op — what the
    # metric reports when the pipeline is handed perfect panels.
    rng0 = np.random.default_rng(999)
    specs0 = make_panel_specs(_CANVAS_H, _CANVAS_W, _GRID, _OVERLAP, rng0)
    merged0 = {}
    for ch in ("H", "S", "O"):
        pristine = [corrupt_panel(skies[ch], PanelSpec(
            s.row0, s.col0, s.height, s.width, 1.0, 0.0,
            s.electrons_per_unit, s.read_sigma), rng0) for s in specs0]
        merged0[ch] = _merge(tmp_path, "sw_floor", ch, pristine, specs0)
    floor = truth_chroma_bias(merged0, skies)["chroma_bias"]
    print(f"[sweep] zero-error floor: {floor:.4f} display units")

    biases = {}
    worst = None
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        specs = make_panel_specs(_CANVAS_H, _CANVAS_W, _GRID, _OVERLAP, rng)
        merged_cur = {}
        for ch in ("H", "S", "O"):
            cur = [corrupt_panel(skies[ch], s, rng) for s in specs]
            merged_cur[ch] = _merge(tmp_path, f"sw{seed}c", ch, cur, specs)
        bias = truth_chroma_bias(merged_cur, skies)["chroma_bias"]
        biases[seed] = bias
        gains = [round(s.gain, 3) for s in specs]
        print(f"[sweep] seed={seed} gains={gains} chroma bias={bias:.4f}")
        if worst is None or bias > worst[1]:
            worst = (seed, bias, merged_cur)

    vals = np.array(list(biases.values()))
    print(f"[sweep] bias over {vals.size} draws (floor {floor:.4f}): "
          f"median={np.median(vals):.4f} max={vals.max():.4f} "
          f"above floor+0.01: {(vals > floor + 0.01).sum()}/{vals.size}")

    seed, bias, cur = worst
    save_png(render_dir / "sweep_worst_current_hubble.png",
             hubble_render(cur["S"], cur["H"], cur["O"]))
    save_png(render_dir / "sweep_worst_baseline_hubble.png",
             hubble_render(merged0["S"], merged0["H"], merged0["O"]))
    print(f"[sweep] worst draw seed={seed} bias={bias:.4f}; renders in "
          f"{render_dir}")

    # Metric sanity only — the sweep is a measurement, not a gate.
    assert np.isfinite(vals).all() and np.isfinite(floor)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_four_panel_mosaic_lab(tmp_path):
    from pix.tasks.gaia_xpsd_query import query_xpsd_stars
    from testing.narrowband_stars import render_narrowband_stars
    from testing.seam_metrics import stretched_chroma_spread
    from testing.synthetic_mosaic import (
        corrupt_panel,
        make_panel_specs,
        make_truth_skies,
        panel_wcs_header,
        write_panel_fits,
    )

    rng = np.random.default_rng(0x40541C)
    radius = float(np.hypot(_CANVAS_W, _CANVAS_H) * _SCALE_DEG / 2) + 0.05
    catalog = query_xpsd_stars(_RA, _DEC, radius, magnitude_high=16.5)
    print(f"\n[lab] catalog {len(catalog['stars'])} stars")

    star_layers = {
        ch: np.clip(render_narrowband_stars(
            _CANVAS_H, _CANVAS_W, _CANVAS_WCS, catalog, ch,
            _star_scale(catalog, ch)), 0, 1.0)
        for ch in ("H", "S", "O")
    }
    skies = make_truth_skies(_CANVAS_H, _CANVAS_W, star_layers, rng)
    specs = make_panel_specs(_CANVAS_H, _CANVAS_W, _GRID, _OVERLAP, rng)
    for i, s in enumerate(specs):
        print(f"[lab] panel{i}: gain={s.gain:.3f} offset={s.offset:.2e} "
              f"e-@ped={s.electrons_per_unit * 0.005:.0f} "
              f"read={s.read_sigma:.1e}")

    corrupted = {ch: [corrupt_panel(skies[ch], s, rng) for s in specs]
                 for ch in ("H", "S", "O")}

    # --- SPFC leg: recover every panel's gain from its stars.
    recovered = {}
    for ch in ("H", "S", "O"):
        scales = []
        for i, (panel, spec) in enumerate(zip(corrupted[ch], specs)):
            f = write_panel_fits(tmp_path / f"panel_{ch}_{i}.fits",
                                 panel, ch, panel_wcs_header(spec, _CANVAS_WCS))
            scales.append(_spfc_scale(
                f, ch, panel_wcs_header(spec, _CANVAS_WCS),
                tmp_path / f"spfc_{ch}_{i}.xisf"))
        recovered[ch] = np.array(scales)
        for i, spec in enumerate(specs):
            got = scales[i] / scales[0]
            want = spec.gain / specs[0].gain
            print(f"[lab:{ch}] panel{i} gain want={want:.4f} got={got:.4f}")
            assert abs(got / want - 1.0) < 0.025, (
                f"[{ch}] panel {i}: SPFC gain {got:.4f} vs injected "
                f"{want:.4f} — recovery broken"
            )

    # --- Chains. `baseline` (uncorrupted panels through the same real op)
    # is the chroma metric's scene floor: per-channel autostretch of
    # genuinely different nebulosity produces region spread with ZERO
    # injected error, so the candidate chain is judged against it.
    from testing.synthetic_mosaic import PanelSpec

    merged_spfc, merged_current, merged_base, naive = {}, {}, {}, {}
    for ch in ("H", "S", "O"):
        norm = [p / (s / recovered[ch][0])
                for p, s in zip(corrupted[ch], recovered[ch])]
        pristine = [
            corrupt_panel(skies[ch], PanelSpec(
                s.row0, s.col0, s.height, s.width, 1.0, 0.0,
                s.electrons_per_unit, s.read_sigma),
                rng)
            for s in specs
        ]
        merged_spfc[ch] = _merge(tmp_path, "spfc", ch,
                                 [n.astype(np.float32) for n in norm], specs)
        merged_current[ch] = _merge(tmp_path, "cur", ch, corrupted[ch], specs)
        merged_base[ch] = _merge(tmp_path, "base", ch, pristine, specs)
        naive[ch] = _naive_overlay(corrupted[ch], specs)

    # --- Seam metric, both orientations, candidate chain asserted against
    # the baseline: strong nebulosity gives the column-profile step a
    # nonzero scene floor (an H ridge registers ~1x RMS with ZERO injected
    # error), so "no seam" means "indistinguishable from the zero-error
    # merge", not an absolute number.
    for ch in ("H", "S", "O"):
        sv, sh = _join_steps(merged_spfc[ch])
        cv, chh = _join_steps(merged_current[ch])
        bv, bh = _join_steps(merged_base[ch])
        print(f"[lab:{ch}] join step spfc=({sv:.2f},{sh:.2f})  "
              f"current=({cv:.2f},{chh:.2f})  base=({bv:.2f},{bh:.2f}) xRMS")
        assert sv < bv + 0.35 and sh < bh + 0.35, (
            f"[{ch}] seam above scene floor after SPFC+offset chain: "
            f"({sv:.2f},{sh:.2f}) vs baseline ({bv:.2f},{bh:.2f}) xRMS"
        )

    # --- Chroma metric: the eye's verdict, referenced to the scene floor.
    chroma_spfc = stretched_chroma_spread(merged_spfc)["chroma_spread"]
    chroma_cur = stretched_chroma_spread(merged_current)["chroma_spread"]
    chroma_base = stretched_chroma_spread(merged_base)["chroma_spread"]
    chroma_ctl = stretched_chroma_spread(naive)["chroma_spread"]
    print(f"[lab] chroma spread: baseline={chroma_base:.4f} "
          f"spfc={chroma_spfc:.4f} current={chroma_cur:.4f} "
          f"control={chroma_ctl:.4f} (display units; ~0.01 above "
          f"baseline = edge of visibility)")

    assert chroma_ctl - chroma_base > 0.05, (
        f"control chroma only {chroma_ctl:.4f} vs baseline "
        f"{chroma_base:.4f} — injected errors invisible to the metric; "
        f"lab is not testing anything"
    )
    assert chroma_spfc - chroma_base < 0.01, (
        f"SPFC+offset chain leaves visible chroma: {chroma_spfc:.4f} vs "
        f"scene baseline {chroma_base:.4f}"
    )
