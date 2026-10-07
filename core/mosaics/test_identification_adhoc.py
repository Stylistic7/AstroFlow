"""Ad-hoc mosaic identification: any overlapping targets on one train are
one mosaic, whatever their layout (2026-10-07); a group that would span
more than the canvas limit is split; another train never joins.

Ray-free: the ops are driven in-process (`.remote` and `ray.get` patched)
on synthetic Targets with real spherical footprints and HEALPix cells.
Runs as a script too: `PYTHONPATH=. python mosaics/test_identification_adhoc.py`.
"""
from __future__ import annotations

import datetime
import inspect
import math

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon

import mosaics.identification as ident
from attributes.astrometric_attributes import healpix_cells_from_poly
from cache.context_ref import ContextRef
from mosaics.rise_time import panel_order
from targets.types import Target
from units import ArcsecondsPerPixel, Degrees, Radians

PANE_W, PANE_H = 352.2 / 60, 235.2 / 60     # 9576 x 6388 px at 2.21"/px, degrees


class _LocalRef(ContextRef):
    def __init__(self, value):
        super().__init__(cache_key=("test", value.name), content_hash=value.content_hash)
        object.__setattr__(self, "_v", value)

    def get(self):
        return self._v


def _footprint(ra, dec, pa):
    """A pane's footprint on the sphere: corners offset from the centre along
    the camera axes, rotated by `pa` about local north (an equatorial mount
    keeps PA to local north everywhere, so a row of panes fans)."""
    c = SkyCoord(ra * u.deg, dec * u.deg)
    ras, decs = [], []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        dx, dy = sx * PANE_W / 2, sy * PANE_H / 2
        k = c.directional_offset_by((np.degrees(np.arctan2(dx, dy)) + pa) * u.deg, np.hypot(dx, dy) * u.deg)
        ras.append(k.ra.deg)
        decs.append(k.dec.deg)
    return SingleSphericalPolygon.from_radec(ras, decs, center=(ra, dec))


def _targets(panes, scale=2.21, start=0):
    t0 = datetime.datetime(2026, 10, 6, tzinfo=datetime.timezone.utc)
    refs = {}
    for i, (name, ra, dec, pa) in enumerate(panes, start):
        poly = _footprint(ra, dec, pa)
        h = f"{i:016x}"
        refs[h] = _LocalRef(Target(
            content_hash=h, name=name, center=(Degrees(ra), Degrees(dec)),
            image_scale=ArcsecondsPerPixel(scale), rotation_angle=Radians(math.radians(pa - 180.0)),
            footprint=poly, healpix_cells=healpix_cells_from_poly(poly), projected_footprints=frozendict(),
            stable_hash=h, oldest_date=t0 + datetime.timedelta(minutes=i),
            newest_date=t0 + datetime.timedelta(minutes=i + 1), stack_ids=(), sensor=(9576, 6388),
        ))
    return refs


def _pane_at(center_ra, center_dec, x, y, pa=270.0, name="p"):
    """A pane whose centre sits (x east, y north) degrees from a centre on
    the tangent plane — what Telescopius draws."""
    c = SkyCoord(center_ra * u.deg, center_dec * u.deg)
    k = c.directional_offset_by(np.degrees(np.arctan2(x, y)) * u.deg, np.hypot(x, y) * u.deg)
    return (name, k.ra.deg, k.dec.deg, pa)


def _tangent_grid(center_ra, center_dec, cols, rows, overlap, pa=270.0):
    sx, sy = PANE_H * (1 - overlap), PANE_W * (1 - overlap)   # PA 270: short side runs E-W, long side N-S
    return [_pane_at(center_ra, center_dec, (col - (cols - 1) / 2) * sx, ((rows - 1) / 2 - row) * sy, pa, f"c{col + 1}r{row + 1}")
            for col in range(cols) for row in range(rows)]


def _raw_overlap():
    return inspect.unwrap(ident.targets_overlap._function)   # past profiling and beartype: no Ray, no cache


def _groups(refs):
    raw = _raw_overlap()
    saved = (ident.targets_overlap.remote, ident.ray.get)
    ident.targets_overlap.remote = lambda *a, **k: raw(*a, **k)
    ident.ray.get = lambda x: x
    try:
        return ident._identify_overlapping_targets(refs)
    finally:
        ident.targets_overlap.remote, ident.ray.get = saved


def test_neighbours_at_dec_69_overlap_and_the_ends_do_not():
    refs = _targets(_tangent_grid(27.4, 69.0, cols=4, rows=1, overlap=0.15))
    raw, hs = _raw_overlap(), list(refs)
    assert all(raw(refs[a], refs[b]) for a, b in zip(hs, hs[1:]))
    assert not raw(refs[hs[0]], refs[hs[-1]])


def test_cassiopeia_grid_at_dec_58_to_69_is_one_mosaic():
    (g,) = _groups(_targets(_tangent_grid(27.4, 63.6, cols=7, rows=3, overlap=0.15)))
    assert len(g) == 21


def test_cygnus_like_grid_is_one_mosaic():
    (g,) = _groups(_targets(_tangent_grid(308.0, 42.0, cols=6, rows=4, overlap=0.10)))
    assert len(g) == 24


def test_grid_straddling_0h_is_one_mosaic_with_a_continuous_panel_order():
    refs = _targets(_tangent_grid(2.0, 63.6, cols=7, rows=3, overlap=0.15))
    (g,) = _groups(refs)
    assert len(g) == 21
    ras = [refs[t].get().center[0] for t in panel_order([(t, float(refs[t].get().center[0])) for t in g])]
    assert ras[0] > 330 and ras[-1] < 30                           # west end first, east end last
    assert all((b - a) % 360.0 < 180.0 for a, b in zip(ras, ras[1:]))   # never a jump back across 0h


def test_off_grid_panes_join():
    """An L of panes, a fill-in pane inside the corner and a pane rotated 45
    degrees over it: one mosaic of six. The grid rules refused all three."""
    sx, sy = PANE_H * 0.85, PANE_W * 0.85
    panes = [_pane_at(100.0, 20.0, x, 0.0, name=f"row{i}") for i, x in enumerate((-sx, 0.0, sx))]
    panes += [_pane_at(100.0, 20.0, sx, -sy, name="down1"), _pane_at(100.0, 20.0, sx, -2 * sy, name="down2")]
    panes += [_pane_at(100.0, 20.0, 0.0, -sy, name="fill")]
    panes += [_pane_at(100.0, 20.0, sx / 2, -sy / 2, pa=315.0, name="rotated")]
    (g,) = _groups(_targets(panes))
    assert len(g) == 7


def test_span_guard_splits_a_row_longer_than_the_canvas_limit():
    refs = _targets(_tangent_grid(180.0, 0.0, cols=18, rows=1, overlap=0.15))   # 17 x 3.3 = 57 deg
    groups = _groups(refs)
    assert sorted(len(g) for g in groups) == [4, 14]
    assert all(ident._span([refs[t].get() for t in g]) <= ident.MAX_MOSAIC_SPAN for g in groups)


def test_another_train_over_the_same_field_never_joins():
    panes = _tangent_grid(100.0, 20.0, cols=2, rows=1, overlap=0.15)
    refs = {**_targets(panes[:1], scale=2.21), **_targets(panes[1:], scale=1.0, start=10)}
    assert _groups(refs) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
