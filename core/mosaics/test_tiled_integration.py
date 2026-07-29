"""End-to-end PIXEL-IDENTITY proof that incremental tile scoping is correct.

This is the definitive test the pure `test_tiled.py` cannot give: it traces a real
`process_in_tiles` graph over a 2-panel Mosaic, drives it through `run_graph`
twice in the SAME `test_context` (so the cache carries over), changing ONE panel's
data between runs, and proves:

  1. PIXEL IDENTITY — the incrementally-recomputed blend is BYTE-IDENTICAL
     (np.array_equal) to an honest full recompute of the changed (v2) state. A
     too-narrow scope would leave a stale v1 tile in the incremental result and
     this fails.
  2. INCREMENTALITY — only the tiles the changed panel touches got a new
     crop_tile key and re-ran; every tile that does NOT overlap the changed panel
     kept its exact crop_tile output path and was a genuine cache HIT (not
     re-executed).

Isolation: uses the conftest `test_context` fixture (a fresh test-mode Ray + a
temp output_dir + a hermetic context), so it never touches the real shared
diskcache. Marked `exhaustive` (opt-in) and given a bounded timeout.

Ephemeral GC: `process_in_tiles` marks the crops/ops ephemeral; `run_graph`
reclaims them after `blend` consumes them, which would delete run 1's crops
before run 2 could hit them. We neutralize the reclaim (no-op
`cleanup_ephemeral_artifact`) ONLY so the cross-run crop cache HIT is observable
at the file level. Pixel identity (assertion 1) is independent of this — a
re-executed crop of an unchanged region produces byte-identical pixels
(content_hash is the raw-pixel hash), so B2 is correct either way.

  PYTHONPATH=. pytest -m exhaustive mosaics/test_tiled_integration.py
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
from file_io.tiling import crop_source_region, feather_blend_to_bigtiff, tile_grid
from file_io.types import TiffFilePath, WithSidecar
from metadata.classes import ImageMetadata
from mosaics.tiled import _panel_scopes, _target_pixel_bboxes, crop_tile, process_in_tiles
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


# A trivial deterministic op: copy the tile through UNCHANGED (no rc-astro), and
# drop a unique marker per EXECUTION so run 2's executed-op count is observable
# (markers survive worker processes; unique names avoid append races).
@node(auto_memory=True, plate_solve=False)
def identity_tile(mosaic: Mosaic, config: ProcessingConfig,
                  image: ImageMetadata) -> WithSidecar[TiffFilePath]:
    out = identity_tile.filename_fn(mosaic, config, image)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    markers = Path(out).parent.parent / "_identity_exec"
    markers.mkdir(parents=True, exist_ok=True)
    (markers / f"{os.getpid()}_{time.time_ns()}_{os.urandom(4).hex()}.marker").write_text(str(out))
    data = np.asarray(read_image(Path(image.filename)), dtype=np.float32)
    write_tiff_image(Path(out), data, {"PIPELINE": "AstroFlow-OSS"})
    return WithSidecar(TiffFilePath(out))


def _tiled_pipeline(mosaic, config, source, dims):
    # The unit under test: tile `source`, run the identity op per tile, feather.
    return process_in_tiles(identity_tile, mosaic, config, source, dims=dims)


def _source_meta(path: Path, w: int, h: int) -> ImageMetadata:
    # Direct construction (compute_image_metadata is a remote @node); the ops only
    # read `.filename`. content_hash is the real file hash (excluded from the
    # crop key on the mosaic path, but set honestly so the contrast is real).
    return ImageMetadata(
        content_hash=get_content_hash(path), stable_hash=str(path.stem),
        filename=path, header={"PIPELINE": "AstroFlow-OSS"},
        width_pixels=w, height_pixels=h, channels=1, dtype=np.dtype("float32"))


def _blend_files(out_dir: Path) -> set:
    return set(out_dir.rglob("blend_tiles/*.tif")) | set(out_dir.rglob("blend_tiles/*.tiff"))


@pytest.mark.exhaustive
@pytest.mark.timeout(900)
def test_incremental_scoping_is_pixel_identical_and_selective(test_context, tmp_path):
    # ---- Build a 2-panel mosaic whose panels project to DISJOINT regions ----
    fp_a = _box(99.40, 99.72, -0.30, 0.30, 99.56, 0.0)   # left interior
    fp_b = _box(100.30, 100.62, -0.30, 0.30, 100.46, 0.0)  # right interior
    fp_m = _box(99.00, 101.00, -0.50, 0.50, 100.00, 0.0)   # spans both + margins
    context.targets = {"A": _target("A", "A_v1", fp_a), "B": _target("B", "B_v1", fp_b)}
    mosaic = _mosaic(fp_m)

    vb = mosaic.viewbox("Master")
    dims = (vb.num_pixels_x, vb.num_pixels_y)
    w, h = dims

    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        tiling=TilingSettings(enabled=True, tile_px=384, overlap=0.0))
    rects = tile_grid(w, h, config.tiling.tile_px, config.tiling.overlap)

    # ---- PREREQUISITE: scoping actually ENGAGED and is SELECTIVE -------------
    scopes_v1 = _panel_scopes(mosaic, dims, rects, config)
    assert all(s is not None for s in scopes_v1), \
        "scoping fell back to the conservative source key — construction is wrong"
    bboxes = dict(_target_pixel_bboxes(mosaic, vb))   # keyed by CONTENT hash
    assert bboxes["A_v1"] is not None and bboxes["B_v1"] is not None
    ax0, ay0, ax1, ay1 = bboxes["A_v1"]
    bx0, by0, bx1, by1 = bboxes["B_v1"]
    assert ax1 < bx0 or bx1 < ax0, f"panels not disjoint in x: {bboxes}"

    a_tiles = [i for i, s in enumerate(scopes_v1) if "A_v1" in s]
    non_a = [i for i, s in enumerate(scopes_v1) if "A_v1" not in s]
    assert 0 < len(a_tiles) < len(rects), \
        f"need A to touch a strict minority of tiles: {len(a_tiles)}/{len(rects)}"
    assert non_a, "need at least one tile the changed panel does NOT touch"
    # There must be a tile that sees ONLY B (proves real spatial separation).
    assert any("B_v1" in scopes_v1[i] and "A_v1" not in scopes_v1[i] for i in range(len(rects))), \
        "no B-only tile — panels are not spatially separated enough"

    # ---- Two sources differing ONLY in panel A's pixel region ---------------
    rng = np.random.default_rng(7)
    base = rng.random((h, w)).astype(np.float32)
    src_v1 = base.copy()
    src_v2 = base.copy()
    cx0, cy0 = max(0, ax0), max(0, ay0)
    cx1, cy1 = min(w, ax1), min(h, ay1)
    src_v2[cy0:cy1, cx0:cx1] = rng.random((cy1 - cy0, cx1 - cx0)).astype(np.float32)
    assert not np.array_equal(src_v1, src_v2)          # A region changed
    # Everything OUTSIDE panel A's bbox is byte-identical between the two sources.
    outside = src_v1.copy(); outside[cy0:cy1, cx0:cx1] = 0
    outside2 = src_v2.copy(); outside2[cy0:cy1, cx0:cx1] = 0
    assert np.array_equal(outside, outside2)

    p_v1 = tmp_path / "source_v1.tiff"
    p_v2 = tmp_path / "source_v2.tiff"
    write_tiff_image(p_v1, src_v1, {"PIPELINE": "AstroFlow-OSS"})
    write_tiff_image(p_v2, src_v2, {"PIPELINE": "AstroFlow-OSS"})
    src1 = _source_meta(p_v1, w, h)
    src2 = _source_meta(p_v2, w, h)
    # The OLD (pre-fix) key kept `source`; its content_hash moves on ANY panel
    # change, which is exactly why every tile used to re-key.
    assert src1.content_hash != src2.content_hash

    out_dir = Path(context.output_dir)

    # crop_tile output paths per tile, per run — the fix's KEY-level claim.
    # scopes are recomputed from the CURRENT context.targets (updated below).
    def crop_paths(source):
        scopes = _panel_scopes(mosaic, dims, rects, config)
        return [crop_tile.filename_fn(mosaic, r, source, s, config)
                for r, s in zip(rects, scopes)], scopes

    paths_v1, _ = crop_paths(src1)

    # ---- Neutralize ephemeral reclaim so cross-run crop HITS are observable --
    import cache.ephemeral_cleanup as ec
    orig_cleanup = ec.cleanup_ephemeral_artifact
    ec.cleanup_ephemeral_artifact = lambda *a, **k: None
    try:
        # RUN 1 (v1) --------------------------------------------------------
        before1 = _blend_files(out_dir)
        run_graph(trace(_tiled_pipeline, mosaic, config, src1, dims))
        b1_files = _blend_files(out_dir) - before1
        if len(b1_files) != 1:
            tree = sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file())
            raise AssertionError(f"expected 1 blend output, got {b1_files}. out_dir tree:\n" + "\n".join(tree))
        b1_path = b1_files.pop()

        # Snapshot crop files (path -> mtime_ns) written by run 1.
        crop_dir = out_dir / "crop_tile"
        crop_mtimes_1 = {str(p): p.stat().st_mtime_ns for p in crop_dir.rglob("*.tiff")}
        exec_dir = out_dir / "_identity_exec"
        execs_after_1 = len(list(exec_dir.glob("*.marker"))) if exec_dir.exists() else 0

        # ---- CHANGE ONE PANEL'S DATA (A) ----------------------------------
        # A's identity (stable_hash / dict key) persists; only its content moves.
        context.targets = {"A": _target("A", "A_v2", fp_a),
                           "B": context.targets["B"]}
        paths_v2, scopes_v2 = crop_paths(src2)

        # KEY-level incrementality: non-A tiles keep their EXACT crop path
        # (source excluded), A tiles get a NEW path.
        changed = [i for i in range(len(rects)) if paths_v1[i] != paths_v2[i]]
        assert sorted(changed) == sorted(a_tiles), \
            f"crop key moved for the wrong tiles: {changed} vs A-tiles {a_tiles}"

        time.sleep(0.05)  # ensure any rewrite would show a newer mtime

        # RUN 2 (v2) — incremental --------------------------------------------
        before2 = _blend_files(out_dir)
        run_graph(trace(_tiled_pipeline, mosaic, config, src2, dims))
        b2_files = _blend_files(out_dir) - before2
        assert len(b2_files) == 1, f"expected 1 new blend output, got {b2_files}"
        b2_incr = b2_files.pop()
    finally:
        ec.cleanup_ephemeral_artifact = orig_cleanup

    # ---- ASSERTION 1: PIXEL IDENTITY vs an honest full recompute ------------
    ref_path = tmp_path / "reference_v2.tiff"
    feather_blend_to_bigtiff(
        [tuple(r) for r in rects],
        # crop directly from source_v2 at every rect (no cache, all-v2 state)
        [_crop_to(tmp_path, i, p_v2, r) for i, r in enumerate(rects)],
        ref_path, overlap=config.tiling.overlap, blend=config.tiling.blend,
        metadata={"PIPELINE": "AstroFlow-OSS"})

    incr = np.asarray(read_image(b2_incr), dtype=np.float32)
    ref = np.asarray(read_image(ref_path), dtype=np.float32)
    assert incr.shape == ref.shape, (incr.shape, ref.shape)
    n_diff = int(np.count_nonzero(incr != ref))
    assert np.array_equal(incr, ref), (
        f"STALE TILE: incremental blend differs from full recompute in "
        f"{n_diff} pixels — a too-narrow scope left a v1 tile in the v2 result")

    # ---- ASSERTION 2: INCREMENTALITY actually happened ----------------------
    # (a) crop_tile file evidence: non-A crop files UNTOUCHED (same path & mtime
    #     => cache hit, not re-executed); A tiles produced NEW crop files.
    non_a_untouched = 0
    for i in non_a:
        p = str(paths_v1[i])
        assert Path(p).exists(), f"non-A crop {p} vanished"
        assert Path(p).stat().st_mtime_ns == crop_mtimes_1[p], \
            f"non-A crop {p} was re-executed (mtime changed) — no incremental hit"
        non_a_untouched += 1
    for i in a_tiles:
        assert Path(paths_v2[i]).exists() and str(paths_v2[i]) not in crop_mtimes_1, \
            f"A-tile crop {paths_v2[i]} was not freshly produced in run 2"

    # (b) executed-op count in run 2: only the A tiles' identity ops ran.
    execs_after_2 = len(list(exec_dir.glob("*.marker")))
    run2_execs = execs_after_2 - execs_after_1
    assert run2_execs == len(a_tiles), \
        f"run 2 executed {run2_execs} ops; expected only the {len(a_tiles)} A-tiles"

    print(
        f"\n[incremental-scoping] tiles={len(rects)} "
        f"A-tiles(re-ran)={len(a_tiles)} non-A(cache HIT)={non_a_untouched} "
        f"run2_op_execs={run2_execs} pixel_diff={n_diff} (0=identical) "
        f"dims={dims} scoping_engaged=True")


def _crop_to(tmp: Path, i: int, source_path: Path, rect) -> str:
    """Write source_path[rect] to a temp tile file; the reference builder feathers
    these directly (an honest full recompute of the v2 tiled result)."""
    p = tmp / f"ref_tile_{i}.tiff"
    write_tiff_image(p, np.asarray(crop_source_region(source_path, rect), dtype=np.float32),
                     {"PIPELINE": "AstroFlow-OSS"})
    return str(p)
