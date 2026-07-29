"""Pure tests for the incremental cache scoping in `mosaics.tiled`.

CORRECTNESS-CRITICAL: a too-narrow tile `panel_scope` silently produces a WRONG
tile, so these lean on the conservative side (margin, straddling panels, "nothing
dropped" property).

Ray-free by construction — they exercise the pure geometry (`_scope_for_rect`,
`tile_grid`, `_target_pixel_bboxes` with a duck viewbox) and the narrow tuple
(`_crop_narrow`), never a real `@node`/Ray call. Run them with PLAIN PYTHON
(`PYTHONPATH=. python mosaics/test_tiled.py`), NOT pytest: the repo's conftest
inits Ray and its teardown does a machine-wide pkill that would kill a live app.
A `__main__` runner at the bottom runs every `test_*` and asserts Ray never
initialized. They are also valid pytest tests if ever run in an isolated venv.
"""
from types import SimpleNamespace

import numpy as np
from spherical_geometry.polygon import SingleSphericalPolygon

from decisions.settings import ProcessingConfig
from file_io.tiling import tile_grid
from geometry.projections import mercator
from geometry.polygons import project_polygon
from hashing.hashing import hash_call
from mosaics.tiled import (
    _crop_narrow,
    _panel_scopes,
    _scope_for_rect,
    _target_pixel_bboxes,
    _viewbox_for_dims,
)


# --------------------------------------------------------------------------- #
# _scope_for_rect — the pure scoping decision                                  #
# --------------------------------------------------------------------------- #

def test_scope_exact_intersection_no_margin():
    # 3 side-by-side 100px tiles; a panel per region + one straddling the
    # tile0/tile1 boundary. Margin 0 == exact overlap.
    rects = tile_grid(300, 100, tile_px=100, overlap=0.0)
    assert rects == [(0, 0, 100, 100), (100, 0, 100, 100), (200, 0, 100, 100)]

    boxes = [
        ("A", (10, 10, 40, 40)),     # tile0 only
        ("B", (90, 10, 110, 40)),    # straddles tile0 <-> tile1
        ("C", (210, 10, 240, 40)),   # tile2 only
    ]
    scope = [_scope_for_rect(r, boxes, always=(), margin=0) for r in rects]
    assert scope[0] == ("A", "B")
    assert scope[1] == ("B",)        # straddling panel in BOTH tiles it touches
    assert scope[2] == ("C",)


def test_scope_margin_pulls_in_feathering_neighbor():
    # Panel D sits in tile0 (50..80), NOT geometrically in tile1 (100..200).
    # With no margin tile1 excludes it; with a margin >= the gap it's pulled in
    # (models a panel that only feathers into the neighbour through a tiled
    # stage). Too broad is safe; the point is never to MISS such a panel.
    rects = tile_grid(300, 100, tile_px=100, overlap=0.0)
    boxes = [("D", (50, 10, 80, 40))]
    tile1 = rects[1]

    assert _scope_for_rect(tile1, boxes, always=(), margin=0) == ()
    assert _scope_for_rect(tile1, boxes, always=(), margin=25) == ("D",)  # 80+25 >= 100


def test_scope_touching_edge_is_included():
    # A panel whose bbox edge exactly meets the tile edge counts as overlapping
    # (rounding toward inclusion).
    rect = (100, 0, 100, 100)  # x in [100, 200]
    boxes = [("E", (60, 10, 100, 40))]  # right edge == rect left edge
    assert _scope_for_rect(rect, boxes, always=(), margin=0) == ("E",)


def test_scope_always_present_panels_in_every_tile():
    # Panels with no finite bbox (the `always` set) must appear in every scope.
    rects = tile_grid(300, 100, tile_px=100, overlap=0.0)
    boxes = [("A", (10, 10, 40, 40))]
    scopes = [_scope_for_rect(r, boxes, always=("GLOBAL",), margin=0) for r in rects]
    assert all("GLOBAL" in s for s in scopes)
    assert scopes[0] == ("A", "GLOBAL")


def test_property_union_of_scopes_is_all_targets():
    # Nothing dropped: with a covering grid every panel lands in >= 1 tile.
    rng = np.random.default_rng(0)
    boxes = []
    for i in range(40):
        x0 = int(rng.integers(0, 900))
        y0 = int(rng.integers(0, 900))
        w = int(rng.integers(5, 120))
        h = int(rng.integers(5, 120))
        boxes.append((f"T{i}", (x0, y0, x0 + w, y0 + h)))

    rects = tile_grid(1000, 1000, tile_px=200, overlap=0.25)
    for margin in (0, 37, 200):
        union = set()
        for r in rects:
            union |= set(_scope_for_rect(r, boxes, always=(), margin=margin))
        assert union == {h for h, _ in boxes}, f"dropped a panel at margin={margin}"


def test_changing_one_target_hash_only_touches_its_tiles():
    # The core incremental claim, at the pure-scope level: moving ONE panel's
    # content_hash changes the scope of exactly the tiles it overlaps.
    rects = tile_grid(300, 100, tile_px=100, overlap=0.0)
    boxes = [
        ("A", (10, 10, 40, 40)),     # tile0 only
        ("B", (90, 10, 110, 40)),    # tile0 + tile1
        ("C", (210, 10, 240, 40)),   # tile2 only
    ]
    before = [_scope_for_rect(r, boxes, always=(), margin=0) for r in rects]

    # A's data changes -> new content_hash "A2"; everything else identical.
    boxes_after = [("A2", (10, 10, 40, 40))] + boxes[1:]
    after = [_scope_for_rect(r, boxes_after, always=(), margin=0) for r in rects]

    assert before[0] != after[0]          # tile0 overlaps A -> re-keys
    assert before[1] == after[1]          # tile1 (only B) unchanged
    assert before[2] == after[2]          # tile2 (only C) unchanged


# --------------------------------------------------------------------------- #
# _crop_narrow — cache-key tuple                                               #
# --------------------------------------------------------------------------- #

_CONFIG = ProcessingConfig()


def _key(args):
    """The real cache key hash_call would compute from a _crop_narrow result."""
    narrowed, kwargs = _crop_narrow(args, {})
    return hash_call("crop_tile_fn_hash", *narrowed, **kwargs)


def test_crop_narrow_non_mosaic_is_byte_for_byte():
    # Generic source (panel_scope None): the narrow drops panel_scope AND config
    # and reproduces the exact pre-scoping key (mosaic, rect, source).
    mosaic, rect, source = "some_mosaic", (0, 0, 100, 100), "some_source"
    narrowed, kwargs = _crop_narrow((mosaic, rect, source, None, _CONFIG), {})
    assert narrowed == (mosaic, rect, source)
    assert kwargs == {}


def test_crop_narrow_mosaic_excludes_source():
    # Two DIFFERENT sources, same mosaic/rect/scope/config -> SAME key (the
    # derived source's whole-image churn no longer re-runs the tile).
    mosaic = SimpleNamespace(stable_hash="MOSAIC_STABLE")
    rect = (0, 0, 100, 100)
    scope = ("A", "B")
    src_a = SimpleNamespace(content_hash="derived_v1")
    src_b = SimpleNamespace(content_hash="derived_v2")
    assert _key((mosaic, rect, src_a, scope, _CONFIG)) == \
        _key((mosaic, rect, src_b, scope, _CONFIG))


def test_crop_narrow_mosaic_reacts_to_scope_and_config():
    mosaic = SimpleNamespace(stable_hash="MOSAIC_STABLE")
    rect = (0, 0, 100, 100)
    src = SimpleNamespace(content_hash="derived")
    base = _key((mosaic, rect, src, ("A", "B"), _CONFIG))

    # A panel in the scope changes -> different key.
    assert _key((mosaic, rect, src, ("A2", "B"), _CONFIG)) != base
    # Config change (tuning a tiled op) -> different key (no stale crop).
    other = ProcessingConfig(tiling=_CONFIG.tiling.__class__(tile_px=12345))
    assert _key((mosaic, rect, src, ("A", "B"), other)) != base
    # Same inputs -> stable key.
    assert _key((mosaic, rect, src, ("A", "B"), _CONFIG)) == base


def test_crop_narrow_key_changes_only_for_overlapping_tiles():
    # End-to-end at the KEY level: a panel-hash change re-keys exactly the tiles
    # whose scope contains it, using the real narrow + hash_call.
    mosaic = SimpleNamespace(stable_hash="MOSAIC_STABLE")
    src = SimpleNamespace(content_hash="derived")
    rects = tile_grid(300, 100, tile_px=100, overlap=0.0)
    boxes = [
        ("A", (10, 10, 40, 40)),
        ("B", (90, 10, 110, 40)),
        ("C", (210, 10, 240, 40)),
    ]
    boxes2 = [("A2", (10, 10, 40, 40))] + boxes[1:]

    def keys(bx):
        return [
            _key((mosaic, r, src, _scope_for_rect(r, bx, (), 0), _CONFIG))
            for r in rects
        ]

    before, after = keys(boxes), keys(boxes2)
    assert before[0] != after[0]   # tile0 overlaps A
    assert before[1] == after[1]
    assert before[2] == after[2]


# --------------------------------------------------------------------------- #
# _target_pixel_bboxes — the real project_to_viewbox reuse                     #
# --------------------------------------------------------------------------- #

def _duck_viewbox(resolution=1e-4):
    """A minimal stand-in carrying only what project_to_viewbox reads. Origin at
    (0,0) radians so a projected radian coord x maps to pixel x/resolution."""
    return SimpleNamespace(
        projection=mercator.MercatorProjection(),
        x_min=0.0, y_min=0.0, resolution=resolution,
    )


def _true_pixel_bounds(footprint, viewbox):
    xs, ys = project_polygon(footprint, viewbox.projection)
    xs = [(x - viewbox.x_min) / viewbox.resolution for x in xs]
    ys = [(y - viewbox.y_min) / viewbox.resolution for y in ys]
    return min(xs), min(ys), max(xs), max(ys)


def test_target_bbox_is_outward_superset_of_projection():
    vb = _duck_viewbox()
    footprint = SingleSphericalPolygon.from_cone(10.0, 20.0, 0.5, steps=32)
    mosaic = SimpleNamespace(
        targets=[SimpleNamespace(content_hash="T", footprint=footprint)]
    )
    (chash, (bx0, by0, bx1, by1)), = _target_pixel_bboxes(mosaic, vb)
    assert chash == "T"

    tminx, tminy, tmaxx, tmaxy = _true_pixel_bounds(footprint, vb)
    # Outward-rounded: the integer box strictly contains the true float bounds.
    assert bx0 <= tminx and by0 <= tminy
    assert bx1 >= tmaxx and by1 >= tmaxy
    assert bx0 == int(np.floor(tminx)) and bx1 == int(np.ceil(tmaxx))


def test_target_bboxes_separated_for_disjoint_cones():
    vb = _duck_viewbox()
    # Two cones far apart on the sky -> disjoint pixel boxes.
    f1 = SingleSphericalPolygon.from_cone(10.0, 20.0, 0.3, steps=24)
    f2 = SingleSphericalPolygon.from_cone(40.0, 20.0, 0.3, steps=24)
    mosaic = SimpleNamespace(targets=[
        SimpleNamespace(content_hash="T1", footprint=f1),
        SimpleNamespace(content_hash="T2", footprint=f2),
    ])
    (_, b1), (_, b2) = _target_pixel_bboxes(mosaic, vb)
    # b1 entirely left of b2 (disjoint on x).
    assert b1[2] < b2[0], (b1, b2)


def test_target_straddling_tile_boundary_hits_both_tiles():
    # A real footprint projected to straddle a tile seam must appear in BOTH
    # tiles' scopes.
    vb = _duck_viewbox(resolution=1e-4)
    footprint = SingleSphericalPolygon.from_cone(10.0, 20.0, 0.4, steps=32)
    mosaic = SimpleNamespace(
        targets=[SimpleNamespace(content_hash="STRD", footprint=footprint)]
    )
    boxes = [(h, b) for h, b in _target_pixel_bboxes(mosaic, vb) if b is not None]
    bx0, by0, bx1, by1 = boxes[0][1]
    seam = (bx0 + bx1) // 2  # a tile boundary through the middle of the panel

    # rects are (x0, y0, tile_w, tile_h); make both tall enough to span the panel.
    ry0, rth = by0 - 100, (by1 - by0) + 200
    left = (seam - 50, ry0, 50, rth)   # x in [seam-50, seam]
    right = (seam, ry0, 50, rth)       # x in [seam, seam+50]
    assert "STRD" in _scope_for_rect(left, boxes, (), margin=0)
    assert "STRD" in _scope_for_rect(right, boxes, (), margin=0)


# --------------------------------------------------------------------------- #
# _panel_scopes / _viewbox_for_dims — composition + generality                 #
# --------------------------------------------------------------------------- #

def test_panel_scopes_non_mosaic_is_all_none():
    # Generality: a non-Mosaic source router -> every tile keeps the pre-scoping
    # (mosaic, rect, source) key. ZERO regression for single-image sources.
    rects = tile_grid(300, 100, tile_px=100, overlap=0.0)
    scopes = _panel_scopes("not_a_mosaic", (300, 100), rects, _CONFIG)
    assert scopes == [None] * len(rects)


def test_viewbox_for_dims_matches_by_pixel_size():
    # Duck mosaic exposing .viewbox(mode); resolution matches only the right mode.
    class _FakeMosaic:
        def viewbox(self, mode):
            sizes = {
                "Subframe": (2048, 2048),
                "DownscaledMaster": (500, 400),
                "Master": (4000, 3200),
            }
            w, h = sizes[mode]
            return SimpleNamespace(num_pixels_x=w, num_pixels_y=h)

    m = _FakeMosaic()
    assert _viewbox_for_dims(m, (4000, 3200)).num_pixels_x == 4000
    assert _viewbox_for_dims(m, (500, 400)).num_pixels_y == 400
    assert _viewbox_for_dims(m, (123, 456)) is None  # no match -> conservative


# --------------------------------------------------------------------------- #
# runner                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import ray  # module import only; we assert it never initializes

    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            import traceback
            failures.append(name)
            print(f"FAIL {name}: {e!r}")
            traceback.print_exc()

    assert not ray.is_initialized(), "Ray must never initialize in these tests"
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed; "
          f"ray_initialized={ray.is_initialized()}")
    raise SystemExit(1 if failures else 0)
