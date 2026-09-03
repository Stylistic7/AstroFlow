"""Pure tests for panels-as-tiles (`mosaics.panels`).

Proves the two load-bearing properties WITHOUT Ray (plain-python geometry + the
real cache-key narrow + hash_call):

  * **No collisions** — no two distinct (panel × stage) tiles ever share a crop
    cache key. This is the regression guard for the mono-stars bug class.
  * **O(change) re-runs** — changing one panel's hash re-keys EXACTLY the tiles
    whose scope contains it (itself + its overlap ring), nothing else.

Ray-free by construction (like `test_tiled.py`): run with PLAIN PYTHON
(`PYTHONPATH=. python mosaics/test_panels.py`), NOT pytest — the repo conftest
inits Ray and its teardown pkills, which would kill a live app. A `__main__`
runner at the bottom runs every `test_*` and asserts Ray never initialized.
"""
from types import SimpleNamespace

import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon

from decisions.settings import ProcessingConfig, TilingSettings
from geometry.projections import mercator
from geometry.polygons import project_polygon
from hashing.hashing import hash_call, hash_list
from mosaics.panels import (
    _crop_panel_narrow,
    _panel_pixel_bboxes,
    _rects_overlap,
    base_panel_mosaic,
    panel_tiles,
)

# Small absolute overlap so the tiny synthetic panels (~150 px bboxes in a ~900 px
# viewbox) stay spatially distinct — the production default (512 px) would pad
# every panel across the whole toy canvas. The geometry/collision/incremental
# properties under test are scale-invariant; only the px value differs.
_CONFIG = ProcessingConfig(tiling=TilingSettings(overlap_px=20))


# --------------------------------------------------------------------------- #
# Scaffolding — a duck viewbox + a mosaic of real spherical-cone panels        #
# --------------------------------------------------------------------------- #

def _viewbox_for(panels, resolution=1e-4, margin_px=200):
    """A viewbox FITTED to the given panels (origin at their projected min, canvas
    sized to contain them + a margin) — mirrors how compute_viewbox sizes a real
    mosaic, so the panels land INSIDE the canvas (not off-edge)."""
    proj = mercator.MercatorProjection()
    xs, ys = [], []
    for p in panels:
        px, py = project_polygon(p.footprint, proj)
        xs += list(px)
        ys += list(py)
    x_min, y_min = min(xs) - margin_px * resolution, min(ys) - margin_px * resolution
    w = int((max(xs) - min(xs)) / resolution) + 2 * margin_px
    h = int((max(ys) - min(ys)) / resolution) + 2 * margin_px
    return SimpleNamespace(
        projection=proj, x_min=x_min, y_min=y_min, resolution=resolution,
        num_pixels_x=w, num_pixels_y=h,
        # The narrow keys on the viewbox; a real one carries its producer's
        # call hash (the pure-Merkle runtime stamp). The duck fakes that so
        # hash_list reduces it instead of address-repring a SimpleNamespace.
        content_hash="VIEWBOX_HASH",
    )


def _panel(pid, content_hash, ra, dec, radius=0.4):
    return SimpleNamespace(
        id=pid, content_hash=content_hash,
        # The crop narrow routes on the panel's STABLE id (content churn rides
        # in scope_hashes) — same convention as production Targets.
        stable_hash=f"stable_{pid}",
        footprint=SingleSphericalPolygon.from_cone(ra, dec, radius, steps=32),
    )


def _mosaic(*panels):
    return SimpleNamespace(stable_hash="MOSAIC_STABLE", targets=list(panels))


def _overlapping_mosaic():
    """Three panels: A and B overlap (centres 0.5° apart, radius 0.4° each →
    0.8° diameter reach); C is far away (disjoint)."""
    return _mosaic(
        _panel("A", "hashA", ra=10.0, dec=20.0),
        _panel("B", "hashB", ra=10.5, dec=20.0),
        _panel("C", "hashC", ra=12.0, dec=20.0),
    )


# --------------------------------------------------------------------------- #
# _rects_overlap — the pure overlap primitive                                  #
# --------------------------------------------------------------------------- #

def test_rects_overlap_cases():
    assert _rects_overlap((0, 0, 10, 10), (5, 5, 15, 15))     # overlapping
    assert _rects_overlap((0, 0, 10, 10), (10, 0, 20, 10))    # edge-touch counts
    assert _rects_overlap((0, 0, 10, 10), (2, 2, 8, 8))       # nested
    assert not _rects_overlap((0, 0, 10, 10), (11, 0, 20, 10))  # b right of a
    assert not _rects_overlap((11, 0, 20, 10), (0, 0, 10, 10))  # b LEFT of a (regression)
    assert not _rects_overlap((0, 0, 10, 10), (0, 11, 10, 20))  # b below a
    assert not _rects_overlap((0, 11, 10, 20), (0, 0, 10, 10))  # b ABOVE a (regression)


# --------------------------------------------------------------------------- #
# Panel geometry — panels ARE the tiling                                       #
# --------------------------------------------------------------------------- #

def test_panel_bbox_is_outward_superset():
    m = _mosaic(_panel("A", "hashA", 10.0, 20.0, radius=0.5))
    vb = _viewbox_for(m.targets)
    (pid, (bx0, by0, bx1, by1)), = _panel_pixel_bboxes(m, vb)
    assert pid == "A"
    xs, ys = project_polygon(m.targets[0].footprint, vb.projection)
    xs = [(x - vb.x_min) / vb.resolution for x in xs]
    ys = [(y - vb.y_min) / vb.resolution for y in ys]
    assert bx0 <= min(xs) and by0 <= min(ys)
    assert bx1 >= max(xs) and by1 >= max(ys)


def test_panel_tiles_one_per_finite_panel_scope_includes_self():
    m = _overlapping_mosaic()
    vb = _viewbox_for(m.targets)
    tiles = panel_tiles(m, vb, _CONFIG)
    assert len(tiles) == 3                                   # one tile per panel
    by_id = {pid: (rect, scope) for pid, rect, scope in tiles}
    for pid, (rect, scope) in by_id.items():
        assert pid in scope                                 # a panel is in its own scope
        x0, y0, w, h = rect
        assert 0 <= x0 and 0 <= y0 and x0 + w <= vb.num_pixels_x and y0 + h <= vb.num_pixels_y


def test_panel_tiles_scope_reflects_overlap():
    m = _overlapping_mosaic()
    vb = _viewbox_for(m.targets)
    scope = {pid: sc for pid, _, sc in panel_tiles(m, vb, _CONFIG)}
    assert "B" in scope["A"] and "A" in scope["B"]          # A, B overlap
    assert "C" not in scope["A"] and "A" not in scope["C"]  # C is disjoint


# --------------------------------------------------------------------------- #
# Cache-key narrow + manifest — collisions & incremental re-keys               #
# --------------------------------------------------------------------------- #

def _key(mosaic, panel, viewbox, source, scope_hashes):
    """The real crop cache-key hash_call computes from a _crop_panel_narrow.
    The rect is no longer IN the key — `(panel, viewbox, pad)` stand in for
    it, computable at trace time when the viewbox is a handle."""
    narrowed, kw = _crop_panel_narrow(
        (mosaic, panel, viewbox, _CONFIG.tiling.overlap_px, source,
         scope_hashes), {})
    return hash_call("crop_panel_fn", *narrowed, **kw)


def _scope_hashes(manifest, scope_ids):
    return tuple(sorted(manifest[j] for j in scope_ids if j in manifest))


def _derive(manifest, op_tag):
    """A stand-in for one process_in_panels stage: bump every panel's hash by an
    op tag (the real code uses `op.filename_fn(...).stem`, but any op-varying,
    scope-folding derivation has the same key-distinctness shape)."""
    return {pid: hash_list([op_tag, h]) for pid, h in manifest.items()}


def test_base_manifest_is_filter_specific():
    m = _overlapping_mosaic()
    mh = base_panel_mosaic(m, "H", merged=None).manifest
    mo = base_panel_mosaic(m, "O", merged=None).manifest
    # Same panels, different filter -> every panel's base hash differs (so the H
    # and O crops can never collide — the latent all-filters bug is closed).
    for pid in ("A", "B", "C"):
        assert mh[pid] != mo[pid]


def test_panel_sources_separate_two_pipelines_on_one_target():
    """Two pipelines processing the SAME panels must not share a colour tail.

    Regression guard for the GraXpert-vs-MGC A/B (2026-08-15): the base manifest
    keyed on `target.content_hash`, which describes the INPUT SUBS and so is
    identical no matter which pipeline processed them. Every scene format
    collided onto one cached product and the two pipelines rendered
    pixel-identical images. `panel_sources` carries the PRE-MERGE product
    identity, which differs, so the tails separate."""
    m = _overlapping_mosaic()
    mgc = {pid: f"mgc_{pid}_stem" for pid in ("A", "B", "C")}
    gxp = {pid: f"gxp_{pid}_stem" for pid in ("A", "B", "C")}

    a = base_panel_mosaic(m, "O", merged=None, panel_sources=mgc).manifest
    b = base_panel_mosaic(m, "O", merged=None, panel_sources=gxp).manifest

    for pid in ("A", "B", "C"):
        assert a[pid] != b[pid], f"panel {pid} collides across pipelines"


def test_panel_source_change_is_panel_local():
    """Changing ONE panel's source moves ONLY that panel's manifest entry.

    This is the property that rules out the tempting one-line fix — folding the
    MERGED channel's whole-image hash into the manifest would also separate the
    pipelines, but every panel would churn whenever any panel moved, costing the
    O(change) re-run guarantee this module exists to provide."""
    m = _overlapping_mosaic()
    before_src = {pid: f"stem_{pid}" for pid in ("A", "B", "C")}
    after_src = dict(before_src, A="stem_A_v2")   # only panel A re-derived

    before = base_panel_mosaic(m, "H", merged=None,
                               panel_sources=before_src).manifest
    after = base_panel_mosaic(m, "H", merged=None,
                              panel_sources=after_src).manifest

    assert before["A"] != after["A"]
    assert before["B"] == after["B"]
    assert before["C"] == after["C"]


def test_panel_sources_are_still_filter_specific():
    """The filter must stay in the hash even with sources supplied, or the H and
    O crops of one panel collide (the all-filters bug the base manifest closed)."""
    m = _overlapping_mosaic()
    src = {pid: f"stem_{pid}" for pid in ("A", "B", "C")}
    h = base_panel_mosaic(m, "H", merged=None, panel_sources=src).manifest
    o = base_panel_mosaic(m, "O", merged=None, panel_sources=src).manifest
    for pid in ("A", "B", "C"):
        assert h[pid] != o[pid]


def test_missing_panel_source_falls_back_to_content_hash():
    """A panel absent from `panel_sources` keeps the pre-2026-08-15 value, so
    callers that never build panels (manifest-algebra unit tests) still work."""
    m = _overlapping_mosaic()
    partial = {"A": "stem_A"}
    got = base_panel_mosaic(m, "H", merged=None,
                            panel_sources=partial).manifest
    plain = base_panel_mosaic(m, "H", merged=None).manifest

    assert got["A"] != plain["A"]      # supplied -> derived from the stem
    assert got["B"] == plain["B"]      # absent   -> content_hash fallback
    assert got["C"] == plain["C"]


def test_no_collisions_across_panels_and_stages():
    m = _overlapping_mosaic()
    vb = _viewbox_for(m.targets)
    tiles = panel_tiles(m, vb, _CONFIG)
    # Simulate the whole tail: H/O base channels, each through BXT then SXT, plus
    # a broadband-style RGB stage — every distinct (panel × stage) crop key.
    manifests = {}
    for filt in ("S", "H", "O", "R", "G", "B"):
        base = base_panel_mosaic(m, filt, merged=None).manifest
        manifests[f"{filt}:base"] = base
        manifests[f"{filt}:bxt"] = _derive(base, "bxt_correct")
        manifests[f"{filt}:sxt"] = _derive(manifests[f"{filt}:bxt"], "sxt_starless")
    by_id = {t.id: t for t in m.targets}
    keys = []
    for stage, manifest in manifests.items():
        for pid, rect, scope in tiles:
            keys.append(_key(m, by_id[pid], vb, None,
                             _scope_hashes(manifest, scope)))
    # Every (panel × stage) crop is a DISTINCT cache key — no product ever
    # collides onto another's crop. (Distinct panels have distinct stable
    # ids; distinct stages have distinct scope hashes.)
    assert len(set(keys)) == len(keys), (
        f"{len(keys) - len(set(keys))} colliding crop keys")


def test_same_panel_same_stage_is_stable_and_dedups():
    m = _overlapping_mosaic()
    vb = _viewbox_for(m.targets)
    tiles = panel_tiles(m, vb, _CONFIG)
    base = base_panel_mosaic(m, "H", merged=None).manifest
    pid, rect, scope = tiles[0]
    panel = next(t for t in m.targets if t.id == pid)
    k1 = _key(m, panel, vb, None, _scope_hashes(base, scope))
    k2 = _key(m, panel, vb, SimpleNamespace(content_hash="ANOTHER_SOURCE"),
              _scope_hashes(base, scope))
    assert k1 == k2   # source excluded from the key -> stable across source churn


def test_only_overlapping_tiles_rekey_on_panel_change():
    m = _overlapping_mosaic()
    vb = _viewbox_for(m.targets)
    tiles = panel_tiles(m, vb, _CONFIG)
    before = base_panel_mosaic(m, "H", merged=None).manifest
    # Panel A re-stacks -> its content_hash moves -> its base manifest hash moves.
    m2 = _mosaic(
        _panel("A", "hashA_v2", 10.0, 20.0),
        _panel("B", "hashB", 10.5, 20.0),
        _panel("C", "hashC", 12.0, 20.0),
    )
    after = base_panel_mosaic(m2, "H", merged=None).manifest

    by_id = {t.id: t for t in m.targets}

    def key_for(pid, manifest):
        rect, scope = next((r, s) for p, r, s in tiles if p == pid)
        return _key(m, by_id[pid], vb, None, _scope_hashes(manifest, scope))

    # SCOPE ALGEBRA IN ISOLATION (merged held fixed at None): A's tile
    # re-keys (its own hash changed); B's re-keys (A ∈ B's scope); C's is
    # unchanged (A ∉ C's scope). In the REAL pipeline a panel change also
    # re-merges, and the manifest's merged-content term then re-keys C too
    # (test_merged_content_change_rekeys_every_tile) — this test pins only
    # the per-panel scoping that gives tiles their identity.
    assert key_for("A", before) != key_for("A", after)
    assert key_for("B", before) != key_for("B", after)
    assert key_for("C", before) == key_for("C", after)


# --------------------------------------------------------------------------- #
# Plain-python runner (NO pytest — see module docstring)                        #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# The merge term — arms must not share tails                                   #
# --------------------------------------------------------------------------- #

def _merged_from(op_source, content="MERGED"):
    """A duck merged-handle the way `present_terminal` stamps one."""
    return SimpleNamespace(content_hash=content, _op_source=op_source)


def test_different_merge_ops_produce_disjoint_manifests():
    """Two pipelines sharing channels but differing only in the MERGE
    algorithm must have disjoint colour tails. Before the merge term, their
    manifests were identical and the whole tail collided onto one set of
    cached products (WBPP-constants vs Montage-planes, 2026-08-26) — the
    first arm to run owned the cache and the other aliased it forever."""
    m = _overlapping_mosaic()
    a = base_panel_mosaic(m, "H", _merged_from("src_oss")).manifest
    b = base_panel_mosaic(m, "H", _merged_from("src_matched")).manifest
    for pid in ("A", "B", "C"):
        assert a[pid] != b[pid]


def test_same_merge_op_produces_identical_manifests():
    """Same algorithm AND same raster content → identical manifests: an
    unchanged mosaic cache-hits its whole tail."""
    m = _overlapping_mosaic()
    a = base_panel_mosaic(m, "H", _merged_from("src_oss")).manifest
    b = base_panel_mosaic(m, "H", _merged_from("src_oss")).manifest
    assert a == b


def test_merged_content_change_rekeys_every_tile():
    """The correctness term (Stylistic7's ruling, 2026-09-03): the level solve
    is GLOBAL — one panel's new subs move every panel's offset, so after a
    re-merge no pixel of the raster is unchanged and EVERY tile must
    re-key. Before this term, distant tiles kept serving crops cut from
    the previous raster (a bounded approximation, retired)."""
    m = _overlapping_mosaic()
    before = base_panel_mosaic(m, "H", _merged_from("src", "raster_v1")).manifest
    after = base_panel_mosaic(m, "H", _merged_from("src", "raster_v2")).manifest
    for pid in ("A", "B", "C"):
        assert before[pid] != after[pid]


def test_merge_source_edit_rekeys_the_tail():
    """`_op_source` is a SOURCE hash: editing the merge algorithm must re-key
    its consumers — before this term, an edited merge rebuilt its rasters
    while every tail product kept serving pixels derived from the previous
    algorithm."""
    m = _overlapping_mosaic()
    before = base_panel_mosaic(m, "H", _merged_from("src_v1")).manifest
    after = base_panel_mosaic(m, "H", _merged_from("src_v2")).manifest
    for pid in ("A", "B", "C"):
        assert before[pid] != after[pid]


if __name__ == "__main__":
    import ray
    fns = {n: f for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)}
    passed = 0
    for name, fn in fns.items():
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            raise
    print(f"\n{passed}/{len(fns)} passed; ray_initialized={ray.is_initialized()}")
    assert not ray.is_initialized(), "Ray must never initialize in this pure test"


