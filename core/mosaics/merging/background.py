"""Panel background LEVEL matching — the missing middle of the mosaic path.

Per-panel background extraction (GraXpert / SPFC+MGC) has an inherent
degeneracy: each panel's model carries an arbitrary additive constant, and
nothing in a per-panel fit ties the panels to a common zero. The result is
panels that are individually flat and collectively mismatched — measured on
the Horsehead (4 panels, 2 strips): R/G background p5–p95 spread 0.00042
against per-pixel σ 0.00013, i.e. large regions sitting at visibly different
pedestals, amplified ~17× by any autostretch whose black point rides just
below sky. The OSS feathered merge hides the seam EDGE but cannot equalise
the LEVELS (see the Apex seam test: PMM removes joins, not pedestal).

The professional fix is global background rectification: solve one
least-squares problem over ALL pairwise overlaps at once so the whole mosaic
is mutually consistent. That solver is Montage's `mBgModel` (IPAC, two
decades of survey service; `brew install montage`), and it is the ONLY part
of Montage used here — the placed panels already share one pixel grid (the
mosaic viewbox), so overlap differencing is plain array math, and Montage's
WCS-driven ends (mProject/mDiff/mFitplane) would only re-derive what the
placement stage already guarantees.

LEVEL-ONLY, deliberately (`mBgModel -l`): GraXpert has already removed each
panel's internal gradient, so the residual disagreement is DC steps — which
is exactly what the Horsehead measurement shows. Solving planes as well
would drag in per-image coordinate conventions for zero measured benefit;
if an A/B ever shows cross-panel gradients surviving, that is the moment to
model slopes, not before.

Ground truth (2026-08-19 synthetic lab, three 400×300 panels, pedestals
0 / +0.010 / −0.005): the full Montage chain recovers pairwise constants to
3 µ (fit c = plus − minus), and applying `corrected = img − correction`
brings all three panels to the same level to 3 µ. The table formats emitted
below are copied verbatim from what mImgtbl/mFitExec wrote in that lab.
"""
# NO `from __future__ import annotations` here, deliberately: it stringifies
# every annotation, and the @node decorator detects file-returning ops by
# INSPECTING the return annotation (`_looks_like_file_return` wants the real
# `WithSidecar[TiffFilePath]`, not the string) — with the future import the
# apply op silently lost its `filename_fn` and every trace of this pipeline
# died with AttributeError at trace time (2026-08-19).

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyvips
import structlog

from context import context  # noqa: F401  (import parity with siblings; unused)
from decorators import node
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic

log = structlog.get_logger()

# Overlap fitting runs on shrunken panels: a DC level estimate does not need
# full resolution, and 8× shrink cuts the I/O 64-fold. Panels share the
# mosaic viewbox grid, so equal shrink keeps them aligned.
_FIT_SHRINK = 8
# A pair must share at least this many (shrunken) pixels to contribute a
# constraint — tiny slivers give noisy constants and mBgModel weights by
# npixel anyway, but a hard floor keeps degenerate corners out entirely.
_MIN_OVERLAP_PX = 500


def _open_shrunk(path: str) -> "np.ndarray":
    img = pyvips.Image.new_from_file(path, access="sequential")
    small = img.shrink(_FIT_SHRINK, _FIT_SHRINK)
    a = np.ndarray(
        buffer=small.write_to_memory(), dtype=np.float32,
        shape=(small.height, small.width, small.bands),
    )
    return a[..., 0] if a.ndim == 3 else a


def pairwise_level_fits(paths: list[str]) -> list[dict]:
    """Median level difference over every overlapping pair.

    Panels are canvas-sized with 0 = no-data (`place_image` fills 0 and
    erodes the edge), so the overlap mask is simply both-nonzero. Median,
    not mean: the overlap contains real structure (stars, nebulosity), and
    the DC disagreement is a property of the SKY.

    Returns Montage-shaped fit rows: `c = plus − minus` (the lab-verified
    convention), plus the npixel/rms fields mBgModel weights by.
    """
    imgs = [_open_shrunk(p) for p in paths]
    fits = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            a, b = imgs[i], imgs[j]
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
            mask = (a != 0) & (b != 0)
            n = int(mask.sum())
            if n < _MIN_OVERLAP_PX:
                continue
            diff = a[mask] - b[mask]
            c = float(np.median(diff))
            mad = float(np.median(np.abs(diff - c)))
            ys, xs = np.nonzero(mask)
            # Real overlap bounding box, back in FULL-RES pixel units.
            # mBgModel validates/weights fits by their box geometry and
            # silently zeroes corrections when the box is degenerate
            # (measured: zeroed-box rows -> all-zero corrections; same rows
            # with real boxes -> exact recovery).
            s = _FIT_SHRINK
            x0, x1 = int(xs.min()) * s, int(xs.max()) * s
            y0, y1 = int(ys.min()) * s, int(ys.max()) * s
            fits.append({
                "plus": i, "minus": j, "c": c,
                "npixel": n * s * s,
                "rms": 1.4826 * mad if mad > 0 else 1e-6,
                "xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1,
                "xcenter": (x0 + x1) / 2.0, "ycenter": (y0 + y1) / 2.0,
                "boxwidth": float(x1 - x0), "boxheight": float(y1 - y0),
            })
    return fits


# ---------------------------------------------------------------------------
# The mBgModel bridge: synthesize the two IPAC tables it reads, run it in
# level-only mode, parse the corrections. Formats copied from the lab run.
# ---------------------------------------------------------------------------

def _write_images_tbl(path: Path, n: int, width: int, height: int) -> None:
    with open(path, "w") as f:
        f.write("| cntr |naxis1|naxis2|     crpix1    |     crpix2    | fname |\n")
        f.write("| int  | int  | int  |     double    |     double    | char  |\n")
        for i in range(n):
            f.write(f"{i:7d}{width:7d}{height:7d}{1.0:16.5f}{1.0:16.5f} p{i} \n")


def _write_fits_tbl(path: Path, fits: list[dict]) -> None:
    cols = ("|     plus|    minus|               a|               b|"
            "               c|        crpix1|        crpix2|      xmin|"
            "      xmax|      ymin|      ymax|      xcenter|      ycenter|"
            "       npixel|             rms|            boxx|            boxy|"
            "        boxwidth|       boxheight|          boxang|\n")
    with open(path, "w") as f:
        f.write(cols)
        for r in fits:
            f.write(
                f"{r['plus']:10d}{r['minus']:10d}{0.0:17.5e}{0.0:17.5e}"
                f"{r['c']:17.5e}{1.0:15.2f}{1.0:15.2f}"
                f"{r['xmin']:11d}{r['xmax']:11d}{r['ymin']:11d}{r['ymax']:11d}"
                f"{r['xcenter']:14.2f}{r['ycenter']:14.2f}{r['npixel']:14d}"
                f"{r['rms']:17.5e}{r['xcenter']:17.1f}{r['ycenter']:17.1f}"
                f"{r['boxwidth']:17.1f}{r['boxheight']:17.1f}{0.0:17.1f} \n")


def _parse_corrections(path: Path, n: int) -> list[float]:
    levels = [0.0] * n
    for line in open(path):
        if line.startswith("|") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 4:
            levels[int(parts[0])] = float(parts[3])
    return levels


def _solve_mbgmodel(fits: list[dict], n: int, width: int, height: int) -> list[float]:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_images_tbl(d / "images.tbl", n, width, height)
        _write_fits_tbl(d / "fits.tbl", fits)
        # -l: level-only (constants, no slopes) — see module docstring.
        r = subprocess.run(
            ["mBgModel", "-l", str(d / "images.tbl"), str(d / "fits.tbl"),
             str(d / "corrections.tbl")],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or 'stat="OK"' not in (r.stdout + r.stderr):
            raise RuntimeError(f"mBgModel failed: {r.stdout} {r.stderr}")
        return _parse_corrections(d / "corrections.tbl", n)


def _solve_lstsq(fits: list[dict], n: int) -> list[float]:
    """Numpy fallback when Montage isn't installed on this node.

    Same problem, min-norm gauge: find per-panel levels L minimising
    Σ w·(L_plus − L_minus − c)² with mean(L) = 0. Kept so a node without
    `brew install montage` still merges seamlessly — but says so, since the
    whole point of this op is wrapping the professional solver.
    """
    rows, rhs, w = [], [], []
    for r in fits:
        row = np.zeros(n)
        row[r["plus"]], row[r["minus"]] = 1.0, -1.0
        rows.append(row)
        rhs.append(r["c"])
        w.append(np.sqrt(r["npixel"]))
    rows.append(np.ones(n))          # gauge: mean level = 0
    rhs.append(0.0)
    w.append(1.0)
    A = np.asarray(rows) * np.asarray(w)[:, None]
    b = np.asarray(rhs) * np.asarray(w)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return [float(x) for x in sol]


@node
def solve_panel_backgrounds(
    mosaic: Mosaic, filter: str, *panels: ImageMetadata,
) -> tuple[float, ...]:
    """Per-panel background levels to SUBTRACT so all panels agree.

    Pure over the placed panels (their content hashes key the cache); the
    result is a handful of floats, so the entry is nearly free and re-solves
    only when a panel actually changes. A single panel — or panels with no
    usable overlaps — correctly gets zero correction: there is nothing to
    disagree with.
    """
    paths = [str(p.filename) for p in panels]
    if len(paths) < 2:
        return tuple(0.0 for _ in paths)
    fits = pairwise_level_fits(paths)
    if not fits:
        log.warning("background matching: no usable overlaps",
                    mosaic=mosaic.stable_hash, filter=filter)
        return tuple(0.0 for _ in paths)
    probe = pyvips.Image.new_from_file(paths[0])
    if shutil.which("mBgModel"):
        levels = _solve_mbgmodel(fits, len(paths), probe.width, probe.height)
    else:
        print("### mBgModel not installed on this node — using the numpy "
              "least-squares fallback (same problem, min-norm gauge). "
              "`brew install montage` for the reference solver.")
        levels = _solve_lstsq(fits, len(paths))
    log.info("background levels solved",
             mosaic=mosaic.stable_hash, filter=filter,
             levels=[round(v, 6) for v in levels],
             pairs=len(fits))
    return tuple(levels)


@node(ephemeral=True, auto_memory=True, plate_solve=False)
def apply_background_level(
    mosaic: Mosaic, filter: str, index: int, *panels: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """Panel `index` with its solved background level subtracted.

    Takes ALL panels (not just its own) because the level is a joint
    property: any panel changing re-solves every level, so every matched
    output must re-key — which the variadic signature encodes directly in
    the call hash. The solve itself runs once and cache-hits for the
    siblings. Streaming end to end: lazy vips arithmetic pulled
    stripe-by-stripe by the tiffsave sink — the multi-GB canvas panel is
    never resident.

    Zero stays zero: 0 is `place_image`'s no-data fill, not sky, so the
    correction applies only where the panel has data.
    """
    from mosaics.color.post_process import _save_mono_vips

    levels = solve_panel_backgrounds(mosaic, filter, *panels)
    level = levels[index]
    src = pyvips.Image.new_from_file(
        str(panels[index].filename), access="sequential")
    if src.bands > 1:
        src = src[0]
    corrected = (src != 0).ifthenelse(src - level, 0.0)
    out = apply_background_level.filename_fn(mosaic, filter, index, *panels)
    _save_mono_vips(str(out), corrected, filter)
    return WithSidecar(TiffFilePath(out))


import ray


@ray.remote(num_cpus=0.1)
def merge_scope_matched(mosaic, filter, viewbox, config, targets, *corrected,
                        dry_run=False):
    """`merge_scope` with ONE inserted step: panel background levels are
    solved globally (mBgModel over every pairwise overlap) and subtracted
    before the OSS blend. Same `delete_intermediates` semantics — the
    placements AND the matched copies are interiors reclaimed at scope
    conclusion; the merge is the terminal and persists. See `merge_scope`
    in mosaics/merging/oss.py for the machinery this mirrors, including the
    runtime fold confession below.
    """
    del dry_run
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace
    from mosaics.merging.oss import merge_images_oss
    from mosaics.placement.astroflow import place_image

    print("### merge_scope_matched fold %s/%s: %s" % (
        getattr(mosaic, "stable_hash", "?")[:12], filter,
        " ".join(str(getattr(c, "content_hash", "?"))[:16] for c in corrected)))

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, config, channel)
            for target, channel in zip(targets, corrected)
        ]
        matched = [
            apply_background_level(mosaic, filter, i, *placed)
            for i in range(len(placed))
        ]
        return merge_images_oss(mosaic, filter, *matched)

    return Graph(trace(_body))
