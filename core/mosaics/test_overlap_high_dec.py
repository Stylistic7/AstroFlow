"""High-declination mosaics must identify like mid-declination ones.

Stylistic7's Cassiopeia grid (2026-10-06, 7x3 RedCat panes at Dec 58-69, PA 270,
15% overlap) grouped as 14 of 21 panes and then failed strip ordering:
`targets_overlap` measured span/thickness as bounding-box extents along a
tangent basis built from a fixed reference vector, which flips from the pole
to the x-axis above |sin Dec| > 0.9 (Dec 64.2 deg) and is then rotated
against the panes' edges, so every stripe looked too thick. The pole
reference is only degenerate AT the pole, so the switch now waits for
|sin Dec| > 0.999 (Dec 87.4 deg); below Dec 64.2 nothing changes.

Ray-free: the ops are driven in-process (`.remote` and `ray.get` patched)
on synthetic Targets with real spherical footprints and HEALPix cells.
Runs as a script too: `PYTHONPATH=. python mosaics/test_overlap_high_dec.py`.
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
from mosaics.overlap import Overlap
from targets.types import Target
from units import ArcsecondsPerPixel, Degrees, Radians

PANE_W, PANE_H = 352.2 / 60, 235.2 / 60     # 9576 x 6388 px at 2.21"/px, degrees
TOL = Radians(math.radians(10.0))


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


def _targets(panes):
    t0 = datetime.datetime(2026, 10, 6, tzinfo=datetime.timezone.utc)
    refs = {}
    for i, (name, ra, dec, pa) in enumerate(panes):
        poly = _footprint(ra, dec, pa)
        h = f"{i:016x}"
        refs[h] = _LocalRef(Target(
            content_hash=h, name=name, center=(Degrees(ra), Degrees(dec)),
            image_scale=ArcsecondsPerPixel(2.21), rotation_angle=Radians(math.radians(89.0)),
            footprint=poly, healpix_cells=healpix_cells_from_poly(poly), projected_footprints=frozendict(),
            stable_hash=h, oldest_date=t0 + datetime.timedelta(minutes=i),
            newest_date=t0 + datetime.timedelta(minutes=i + 1), stack_ids=(),
        ))
    return refs


def _tangent_grid(center_ra, center_dec, cols, rows, overlap, pa=270.0):
    """Panes laid on a tangent-plane grid about the centre (what Telescopius
    draws): rectilinear in the plane, so columns curve on the sky."""
    c = SkyCoord(center_ra * u.deg, center_dec * u.deg)
    sx, sy = PANE_H * (1 - overlap), PANE_W * (1 - overlap)   # PA 270: short side runs E-W, long side N-S
    panes = []
    for col in range(cols):
        for row in range(rows):
            x = (col - (cols - 1) / 2) * sx
            y = ((rows - 1) / 2 - row) * sy
            k = c.directional_offset_by(np.degrees(np.arctan2(x, y)) * u.deg, np.hypot(x, y) * u.deg)
            panes.append((f"c{col + 1}r{row + 1}", k.ra.deg, k.dec.deg, pa))
    return panes


def _run_in_process(refs):
    raw = inspect.unwrap(ident.targets_overlap._function)   # past profiling and beartype: no Ray, no cache
    saved = (ident.targets_overlap.remote, ident.ray.get)
    ident.targets_overlap.remote = lambda *a, **k: raw(*a, **k)
    ident.ray.get = lambda x: x
    try:
        groups = ident._identify_overlapping_targets(refs, TOL)
        strips = {id(g): ident.find_strips(refs, list(g), TOL) for g in groups if len(g) > 1}
        return groups, strips
    finally:
        ident.targets_overlap.remote, ident.ray.get = saved


def _check_grid(panes, cols, rows):
    refs = _targets(panes)
    groups, strips = _run_in_process(refs)
    assert len(groups) == 1 and len(groups[0]) == cols * rows, [len(g) for g in groups]
    (s,) = strips.values()
    assert len(s) == rows and all(len(strip) == cols for strip in s), [len(x) for x in s]


def test_cassiopeia_grid_at_dec_58_to_69_identifies_whole():
    _check_grid(_tangent_grid(27.4, 63.6, cols=7, rows=3, overlap=0.15), 7, 3)


def test_cygnus_like_grid_at_dec_35_to_49_still_identifies():
    _check_grid(_tangent_grid(308.0, 42.0, cols=6, rows=4, overlap=0.10), 6, 4)


def test_neighbours_at_dec_69_are_long_edge_overlaps():
    """The pairs the old basis rejected: a top row at Dec 69, every E-W pair."""
    panes = _tangent_grid(27.4, 69.0, cols=4, rows=1, overlap=0.15)
    refs = _targets(panes)
    raw = inspect.unwrap(ident.targets_overlap._function)
    hs = list(refs)
    for a, b in zip(hs, hs[1:]):
        assert raw(refs[a], refs[b], angular_tolerance=TOL) is Overlap.LONG_EDGE


if __name__ == "__main__":
    for fn in (test_neighbours_at_dec_69_are_long_edge_overlaps,
               test_cassiopeia_grid_at_dec_58_to_69_identifies_whole,
               test_cygnus_like_grid_at_dec_35_to_49_still_identifies):
        fn()
        print("ok", fn.__name__)
