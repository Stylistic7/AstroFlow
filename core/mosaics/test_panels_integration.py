"""End-to-end PIXEL-IDENTITY proof that panels-as-tiles incremental recompute is
correct.

Traces a real `process_in_panels` graph over a 2-panel Mosaic, drives it through
`run_graph` TWICE in the same `test_context` (cache carries over), changing ONE
panel's data between runs, and proves:

  1. PIXEL IDENTITY — the incrementally-recomputed blend is BYTE-IDENTICAL
     (np.array_equal) to an honest full recompute of the changed (v2) state. A
     stale (un-recomputed) panel tile would fail this.
  2. INCREMENTALITY — only the changed panel's tile got a new `crop_panel` key
     and re-ran; the disjoint panel kept its exact `crop_panel` output path and
     was a genuine cache HIT (unchanged file mtime), never re-executed.

This is the run-graph counterpart to the pure-key `test_panels.py`. Marked
`exhaustive`; uses `identity_tile` (no gpu_slots) so run_graph completes (the
color pipeline's gpu_slots ops stall the mock-mode scheduler — unrelated).

  PYTHONPATH=. pytest -m exhaustive mosaics/test_panels_integration.py
"""
import datetime
import os
import time
from pathlib import Path

import numpy as np
import pytest
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from attributes.astrometric_attributes import healpix_cells_from_poly
from context import context
from decisions.settings import DevelopmentSettings, ProcessingConfig, TilingSettings
from decorators import node
from dependency_graph import run_graph
from dependency_graph.tracing import trace
from file_io.generic import get_content_hash, read_image
from file_io.tiff_file_io import write_tiff_image
from file_io.tiling import crop_source_region, feather_blend_to_bigtiff
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.panels import (
    base_panel_mosaic, blend_panels, crop_panel, panel_tiles,
    process_in_panels, _panel_pixel_bboxes,
)
from mosaics.types import Mosaic
from targets.types import Target
from units import ArcsecondsPerPixel, Degrees, Radians

_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_SCALE = ArcsecondsPerPixel(3.6)


def _box(ra0, ra1, dec0, dec1, cra, cdec):
    return SingleSphericalPolygon.from_radec(
        [ra0, ra1, ra1, ra0], [dec0, dec0, dec1, dec1], center=[cra, cdec])


def _target(name: str, content_hash: str, footprint) -> Target:
    return Target(
        name=name, content_hash=content_hash, stable_hash=name,
        oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(100.0), Degrees(0.0)), image_scale=_SCALE,
        rotation_angle=Radians(0.0), footprint=footprint,
        healpix_cells=healpix_cells_from_poly(footprint),
        projected_footprints=frozendict(), stack_ids=(name,))


def _mosaic(footprint) -> Mosaic:
    return Mosaic(
        name="incr-test", content_hash="cm", stable_hash="MOSAIC_INCR",
        target_ids=("A", "B"), strips=(("A",), ("B",)), panel_target_ids=("A", "B"),
        oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(100.0), Degrees(0.0)), image_scale=_SCALE,
        rotation_angle=Radians(0.0), footprint=footprint,
        healpix_cells=healpix_cells_from_poly(footprint),
        projected_footprints=frozendict())


@node(auto_memory=True, plate_solve=False)
def identity_tile(mosaic: Mosaic, config: ProcessingConfig,
                  image: ImageMetadata) -> WithSidecar[TiffFilePath]:
    """Copy the panel tile through UNCHANGED, dropping a unique per-EXECUTION
    marker so run 2's executed-op count is observable."""
    out = identity_tile.filename_fn(mosaic, config, image)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    markers = Path(out).parent.parent / "_identity_exec"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / f"{os.getpid()}_{time.time_ns()}_{os.urandom(4).hex()}.marker").write_text(str(out))
    write_tiff_image(Path(out), np.asarray(read_image(Path(image.filename)), dtype=np.float32),
                     {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _panel_pipeline(mosaic, config, src, viewbox):
    """Tile `src` by PANEL, run identity per panel, feather back → the merged handle."""
    pm = base_panel_mosaic(mosaic, "L", src)
    return process_in_panels(identity_tile, mosaic, config, pm, viewbox=viewbox).merged


def _source_meta(path: Path, w: int, h: int) -> ImageMetadata:
    return ImageMetadata(
        content_hash=get_content_hash(path), stable_hash=str(path.stem),
        filename=path, header={"PIPELINE": "AstroFlow-OSS"},
        width_pixels=w, height_pixels=h, channels=1, dtype=np.dtype("float32"))


def _crop_to(tmp_path: Path, i: int, src_path: Path, rect) -> Path:
    p = tmp_path / f"ref_tile_{i}.tiff"
    write_tiff_image(p, crop_source_region(src_path, rect), {"PIPELINE": "AstroFlow-OSS"})
    return p


def _blend_files(out_dir: Path) -> set:
    return set(out_dir.rglob("blend_panels/*.tif")) | set(out_dir.rglob("blend_panels/*.tiff"))


@pytest.mark.exhaustive
@pytest.mark.timeout(900)
def test_panel_incremental_is_pixel_identical_and_selective(test_context, tmp_path):
    # ---- Build a 2-panel mosaic whose panels project to DISJOINT regions ----
    fp_a = _box(99.40, 99.72, -0.30, 0.30, 99.56, 0.0)     # left
    fp_b = _box(100.30, 100.62, -0.30, 0.30, 100.46, 0.0)  # right
    fp_m = _box(99.00, 101.00, -0.50, 0.50, 100.00, 0.0)   # spans both + margins
    context.targets = {"A": _target("A", "A_v1", fp_a), "B": _target("B", "B_v1", fp_b)}
    mosaic = _mosaic(fp_m)
    vb = mosaic.viewbox("Master")
    w, h = vb.num_pixels_x, vb.num_pixels_y

    # overlap_px=0 → each panel's tile is exactly its bbox (no padding into
    # neighbours), so the disjoint panels have disjoint scopes — the cleanest
    # incremental case (B is provably untouched by an A change).
    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        tiling=TilingSettings(overlap_px=0))

    # ---- PREREQUISITE: the two panels are spatially disjoint -----------------
    tiles = panel_tiles(mosaic, vb, config)
    assert [pid for pid, _, _ in tiles] == ["A", "B"], tiles
    scope = {pid: sc for pid, _, sc in tiles}
    assert scope["A"] == ("A",) and scope["B"] == ("B",), \
        f"panels not disjoint: {scope}"
    bboxes = dict(_panel_pixel_bboxes(mosaic, vb))
    ax0, ay0, ax1, ay1 = bboxes["A"]
    bx0, by0, bx1, by1 = bboxes["B"]
    assert ax1 < bx0 or bx1 < ax0, f"panels not disjoint in x: {bboxes}"

    # ---- Two sources differing ONLY in panel A's pixel region ---------------
    rng = np.random.default_rng(7)
    base = rng.random((h, w)).astype(np.float32)
    src_v1 = base.copy()
    src_v2 = base.copy()
    cx0, cy0, cx1, cy1 = max(0, ax0), max(0, ay0), min(w, ax1), min(h, ay1)
    src_v2[cy0:cy1, cx0:cx1] = rng.random((cy1 - cy0, cx1 - cx0)).astype(np.float32)
    assert not np.array_equal(src_v1, src_v2)
    p_v1, p_v2 = tmp_path / "source_v1.tiff", tmp_path / "source_v2.tiff"
    write_tiff_image(p_v1, src_v1, {"PIPELINE": "AstroFlow-OSS"})
    write_tiff_image(p_v2, src_v2, {"PIPELINE": "AstroFlow-OSS"})
    src1, src2 = _source_meta(p_v1, w, h), _source_meta(p_v2, w, h)
    assert src1.content_hash != src2.content_hash

    out_dir = Path(context.output_dir)
    rects = {pid: rect for pid, rect, _ in tiles}

    def crop_paths() -> dict:
        """{panel_id: crop_panel output path} for the CURRENT context.targets —
        keyed on the per-panel manifest, source EXCLUDED."""
        pm = base_panel_mosaic(mosaic, "L", src1)   # src excluded from key
        pad = config.tiling.overlap_px
        paths = {}
        for pid, rect, sc in panel_tiles(mosaic, vb, config):
            sh = tuple(sorted(pm.manifest[j] for j in sc if j in pm.manifest))
            # The rect left the key; (panel, viewbox, pad) stand in for it.
            paths[pid] = crop_panel.filename_fn(
                mosaic, context.targets[pid], vb, pad, pm.merged, sh)
        return paths

    paths_v1 = crop_paths()

    # Crops persist here: this pipeline builds them in the caller's graph, not
    # inside a `delete_intermediates=True` subgraph, so cross-run crop HITS
    # are directly observable.

    # RUN 1 (v1) --------------------------------------------------------------
    before1 = _blend_files(out_dir)
    run_graph(trace(_panel_pipeline, mosaic, config, src1, vb))
    b1_files = _blend_files(out_dir) - before1
    assert len(b1_files) == 1, f"expected 1 blend output, got {b1_files}"

    # The TILE is the incrementality signal, not the crop: crops are
    # scope-local now (each {crop, op} pair is a subgraph built with
    # `delete_intermediates=True`, so the crop is deleted the moment its op
    # completes). The tile is what escapes that subgraph and persists.
    tile_dir = out_dir / "identity_tile"
    tile_mtimes_1 = {str(p): p.stat().st_mtime_ns for p in tile_dir.rglob("*.tiff")}
    assert len(tile_mtimes_1) == 2, f"expected 2 tiles after run 1, got {tile_mtimes_1}"

    # ---- CHANGE ONE PANEL'S DATA (A) ---------------------------------------
    context.targets = {"A": _target("A", "A_v2", fp_a), "B": context.targets["B"]}
    paths_v2 = crop_paths()
    # KEY-level incrementality: A's crop key MOVED, B's did NOT.
    assert paths_v2["A"] != paths_v1["A"], "A's crop key did not move on re-stack"
    assert paths_v2["B"] == paths_v1["B"], "B's crop key moved (disjoint panel churned)"

    time.sleep(0.05)

    # RUN 2 (v2) — incremental ------------------------------------------------
    before2 = _blend_files(out_dir)
    run_graph(trace(_panel_pipeline, mosaic, config, src2, vb))
    b2_files = _blend_files(out_dir) - before2
    assert len(b2_files) == 1, f"expected 1 new blend output, got {b2_files}"
    b2_incr = b2_files.pop()

    # ---- ASSERTION 1: PIXEL IDENTITY vs an honest full recompute -----------
    ref_path = tmp_path / "reference_v2.tiff"
    ordered = [(pid, rects[pid]) for pid, _, _ in tiles]
    feather_blend_to_bigtiff(
        [tuple(r) for _, r in ordered],
        [_crop_to(tmp_path, i, p_v2, r) for i, (_, r) in enumerate(ordered)],
        ref_path, taper_px=config.tiling.overlap_px, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS"})
    incr = np.asarray(read_image(b2_incr), dtype=np.float32)
    ref = np.asarray(read_image(ref_path), dtype=np.float32)
    assert incr.shape == ref.shape, (incr.shape, ref.shape)
    n_diff = int(np.count_nonzero(incr != ref))
    assert np.array_equal(incr, ref), (
        f"STALE PANEL: incremental blend differs from full recompute in {n_diff} "
        f"pixels — a changed panel left a v1 tile in the v2 result")

    # ---- ASSERTION 2: INCREMENTALITY — only panel A re-ran -----------------
    # `identity_tile` drops one marker per EXECUTION, so re-runs are directly
    # countable: A and B in run 1, then A alone in run 2.
    execs = len(list((out_dir / "_identity_exec").glob("*.marker")))
    assert execs == 3, (
        f"expected 3 identity_tile executions (A+B in run 1, A only in run 2), "
        f"got {execs} — run 2 re-processed a panel whose data didn't change")

    tiles_after = {str(p): p.stat().st_mtime_ns
                   for p in tile_dir.rglob("*.tiff")}
    # Tiles PERSIST (only crops are scope-local), so run 1's two tiles must
    # both still be byte-untouched: B's because it wasn't re-processed, A's
    # because its v1 output is merely superseded, never rewritten in place.
    # If B had been re-run, its path would be rewritten and its mtime would
    # move — so "both untouched" is exactly the no-redundant-work claim.
    survived = [p for p, mt in tile_mtimes_1.items()
                if tiles_after.get(p) == mt]
    assert len(survived) == 2, (
        f"a run-1 tile was rewritten in run 2 — panel B was re-processed "
        f"needlessly; untouched={survived}")
    new_tiles = set(tiles_after) - set(tile_mtimes_1)
    assert len(new_tiles) == 1, (
        f"expected exactly one NEW tile (A's, re-keyed), got {new_tiles}")

    # The crop KEYS still move correctly even though the files are transient —
    # pure `filename_fn` arithmetic, asserted above (paths_v1 vs paths_v2).
    assert not (out_dir / "crop_panel").exists() or not list(
        (out_dir / "crop_panel").rglob("*.tiff")
    ), "scope-local crops should have been deleted when their subgraphs concluded"
