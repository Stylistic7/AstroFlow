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

from decisions.settings import ProcessingConfig
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

_CONFIG = ProcessingConfig()


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
    )


def _panel(pid, content_hash, ra, dec, radius=0.4):
    return SimpleNamespace(
        id=pid, content_hash=content_hash,
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

def _key(mosaic, rect, source, scope_hashes):
    """The real crop cache-key hash_call computes from a _crop_panel_narrow."""
    narrowed, kw = _crop_panel_narrow((mosaic, rect, source, scope_hashes), {})
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
    keys = []
    for stage, manifest in manifests.items():
        for pid, rect, scope in tiles:
            keys.append(_key(m, rect, None, _scope_hashes(manifest, scope)))
    # Every (panel × stage) crop is a DISTINCT cache key — no product ever
    # collides onto another's crop. (Distinct panels have distinct rects;
    # distinct stages have distinct scope hashes.)
    assert len(set(keys)) == len(keys), (
        f"{len(keys) - len(set(keys))} colliding crop keys")


def test_same_panel_same_stage_is_stable_and_dedups():
    m = _overlapping_mosaic()
    vb = _viewbox_for(m.targets)
    tiles = panel_tiles(m, vb, _CONFIG)
    base = base_panel_mosaic(m, "H", merged=None).manifest
    pid, rect, scope = tiles[0]
    k1 = _key(m, rect, None, _scope_hashes(base, scope))
    k2 = _key(m, rect, SimpleNamespace(content_hash="ANOTHER_SOURCE"),
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

    def key_for(pid, manifest):
        rect, scope = next((r, s) for p, r, s in tiles if p == pid)
        return _key(m, rect, None, _scope_hashes(manifest, scope))

    # A's tile re-keys (its own hash changed); B's tile re-keys (A ∈ B's scope);
    # C's tile is UNCHANGED (A ∉ C's scope) -> a genuine cache hit.
    assert key_for("A", before) != key_for("A", after)
    assert key_for("B", before) != key_for("B", after)
    assert key_for("C", before) == key_for("C", after)


# --------------------------------------------------------------------------- #
# Plain-python runner (NO pytest — see module docstring)                        #
# --------------------------------------------------------------------------- #

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
