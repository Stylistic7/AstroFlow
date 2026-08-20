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

FUSED INTO THE MERGE, not a separate graph stage: `merge_images_matched`
(below) is `merge_images_oss` with `level_match=True` — the blend already
reads every bbox-on-disk panel once and intersects every overlapping pair
for its flux solve, so the level fit rides that same pass and the solve is
a plain function call inside the node. Rectify, then feather. Two designs
died before this one (2026-08-20): per-panel apply nodes calling the solve
node at runtime (a @node is a remote task — forbidden), then a levels-as-
1xN-TIFF artifact edge (works, but smuggles floats through a file and
re-reads panels a second time — and its canvas-aligned pairwise fitter was
silently WRONG for bbox-on-disk panels).

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

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import structlog

from context import context  # noqa: F401  (import parity with siblings; unused)
from decorators import node
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.types import Mosaic

log = structlog.get_logger()


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


def solve_levels(rows: list[dict], n: int,
                 canvas_shape: tuple[int, int]) -> list[float]:
    """Global level solve over Montage-shaped fit rows — a PLAIN function,
    called from inside the fused merge (`_blend_many(level_match=True)`).
    mBgModel when installed; the numpy least-squares fallback is the same
    problem in the min-norm gauge. A single panel — or panels with no
    usable overlaps — correctly gets zero correction: there is nothing to
    disagree with."""
    if n < 2 or not rows:
        if n >= 2:
            log.warning("background matching: no usable overlaps")
        return [0.0] * n
    ch, cw = canvas_shape
    if shutil.which("mBgModel"):
        levels = [float(v) for v in _solve_mbgmodel(rows, n, cw, ch)]
    else:
        print("### mBgModel not installed on this node — using the numpy "
              "least-squares fallback (same problem, min-norm gauge). "
              "`brew install montage` for the reference solver.")
        levels = [float(v) for v in _solve_lstsq(rows, n)]
    log.info("background levels solved",
             levels=[round(v, 6) for v in levels], pairs=len(rows))
    return levels


@node(display="Mosaic", final=True, auto_memory=True, plate_solve=False)  # mosaic projection — non-square pixels
def merge_images_matched(
    mosaic: Mosaic, filter: str, *inputs: ImageMetadata,
) -> WithSidecar[TiffFilePath]:
    """`merge_images_oss` with Montage level matching fused in — rectify,
    then feather. The blend already reads every panel once and intersects
    every overlapping pair for its flux solve; the level fit rides that
    same pass (`_level_fit_rows` → `solve_levels` → in-place subtraction),
    so rectification costs no extra I/O and no intermediate artifacts, and
    the solve is a normal function call inside this one node.

    The wrapper mirrors merge_images_oss line for line instead of sharing
    a helper: the twin's body is hash-frozen — extracting one would re-key
    every existing mosaic merge in the parent pipelines."""
    from mosaics.merging.oss import (
        _blend_many, _panel_count, _set_panel_count, write_tiff_image,
    )
    from hashing import hash_list

    assert len(inputs) >= 1
    output_filename = merge_images_matched.filename_fn(mosaic, filter, *inputs)
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)

    paths = [inp.filename.resolve(strict=True) for inp in inputs]
    result = _blend_many(paths, level_match=True)

    write_tiff_image(Path(output_filename), result.astype(np.float32),
                     {"FILTER": filter,
                      "STBLHASH": hash_list([inp.stable_hash for inp in inputs]),
                      "PIPELINE": "AstroFlow-OSS-Montage"})

    new_count = sum(_panel_count(inp) for inp in inputs)
    _set_panel_count(output_filename, new_count)

    log.info("level-matched OSS merge complete", output=str(output_filename),
             panel_count=new_count, n_inputs=len(inputs))
    return WithSidecar(output_filename)


import ray


@ray.remote(num_cpus=0.1)
def merge_scope_matched(mosaic, filter, viewbox, config, targets, *corrected,
                        dry_run=False):
    """`merge_scope` with its terminal swapped: `merge_images_matched`
    fuses the global level solve (mBgModel over every pairwise overlap)
    into the OSS blend itself. Same shape, same `delete_intermediates`
    semantics — the placements are interiors reclaimed at scope
    conclusion; the merge is the terminal and persists. See `merge_scope`
    in mosaics/merging/oss.py for the machinery this mirrors, including
    the runtime fold confession below.
    """
    del dry_run
    from dependency_graph.classes import Graph
    from dependency_graph.tracing import trace
    from mosaics.placement.astroflow import place_image

    print("### merge_scope_matched fold %s/%s: %s" % (
        getattr(mosaic, "stable_hash", "?")[:12], filter,
        " ".join(str(getattr(c, "content_hash", "?"))[:16] for c in corrected)))

    def _body():
        placed = [
            place_image(mosaic, target, filter, viewbox, config, channel)
            for target, channel in zip(targets, corrected)
        ]
        return merge_images_matched(mosaic, filter, *placed)

    return Graph(trace(_body))
