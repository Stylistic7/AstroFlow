"""Integration test for OSS mosaic color post-processing.

Traces `color_post_process` on synthetic per-filter channels and runs it through
the executor (mock path — no rc-astro), asserting the named color deliverables
(Hubble, Foraxx, Stars) are produced as valid 3-channel images, and that the
workflow_group wires the expected per-channel ops. Plus direct glue tests of the
palette/screen file I/O (with and without a stars layer).
"""
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

import astropy.io.fits as pyfits

from attributes.astrometric_attributes import healpix_cells_from_poly
from context import context
from decisions.settings import DevelopmentSettings, ProcessingConfig
from dependency_graph import run_graph
from dependency_graph.classes import Node
from dependency_graph.tracing import trace
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from metadata.classes import ImageMetadata
from mosaics.color import post_process as pp
from mosaics.types import Mosaic
from units import ArcsecondsPerPixel, Degrees, Radians

_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_FOOTPRINT = SingleSphericalPolygon.from_radec(
    [98.0, 102.0, 102.0, 98.0], [-1, -1, 1, 1], center=[100.0, 0]
)
_HEALPIX = healpix_cells_from_poly(_FOOTPRINT)


def _write_channel(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.05, np.float32)
    img += 0.2 * np.exp(-(((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 30.0 ** 2)))
    for _ in range(15):
        cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        img += rng.uniform(0.2, 0.8) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
        )
    pyfits.PrimaryHDU(np.clip(img, 0, 1).astype(np.float32)).writeto(path, overwrite=True)


def _mosaic() -> Mosaic:
    return Mosaic(
        name="color-test", content_hash="content_color", stable_hash="mosaic_color",
        target_ids=("a",), strips=(("a",),), panel_target_ids=("a",),
        oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(0.0), Degrees(0.0)), image_scale=ArcsecondsPerPixel(1.0),
        rotation_angle=Radians(0.0), footprint=_FOOTPRINT, healpix_cells=_HEALPIX,
        projected_footprints=frozendict(),
    )


def _count_by_op(graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[int] = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if isinstance(n, Node) and id(n) not in seen:
                seen.add(id(n))
                name = getattr(n.task, "__name__", str(n.task))
                counts[name] = counts.get(name, 0) + 1
    return counts


def test_color_post_process_traces_and_runs(test_context, tmp_path):
    channels = {}
    for i, f in enumerate(("S", "H", "O", "R", "G", "B")):
        p = tmp_path / f"chan_{f}.fits"
        _write_channel(p, i)
        # Construct the input metadata directly (compute_image_metadata is a
        # remote @node, callable only inside a graph); the ops only read
        # `.filename`, and `content_hash` keys the cache.
        channels[f] = ImageMetadata(
            content_hash=f"chan_{f}", stable_hash=f"chan_{f}", filename=p,
            header={"FILTER": f}, width_pixels=128, height_pixels=128,
            channels=1, dtype=np.dtype("float32"),
        )

    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
    )
    graph = trace(pp.color_post_process, _mosaic(), channels, config)
    assert graph is not None

    counts = _count_by_op(graph)
    assert counts.get("bxt_correct") == 6        # one per filter
    assert counts.get("sxt_starless") == 3       # H, O, S (narrowband, per-channel)
    assert counts.get("autostretch_channel") == 3  # narrowband only (H, O, S)
    # Broadband stars: combine R+G+B into one linear RGB, then a SINGLE SXT pass
    # (coherent stars) — not per-channel.
    assert counts.get("combine_broadband_rgb") == 1
    assert counts.get("sxt_stars_rgb") == 1
    assert counts.get("sxt_stars") is None       # old per-channel path removed
    assert counts.get("combine_star_rgb") == 1
    assert counts.get("hubble_image") == 1
    assert counts.get("foraxx_image") == 1
    assert counts.get("hubble_starless") == 1
    assert counts.get("foraxx_starless") == 1

    run_graph(graph)

    out = Path(context.output_dir)
    for op, name in (("hubble_image", "Hubble"),
                     ("foraxx_image", "Foraxx"),
                     ("combine_star_rgb", "Stars")):
        files = list(out.rglob(f"{op}/*.tif")) + list(out.rglob(f"{op}/*.tiff"))
        assert files, f"no {name} output produced under {op}/"
        img = read_image(files[0])
        assert img.ndim == 3 and img.shape[2] == 3, f"{name} not 3-channel: {img.shape}"
        assert np.all(np.isfinite(img))
        assert img.min() >= 0.0 and img.max() <= 1.0001


def test_palette_glue_with_and_without_stars(tmp_path):
    metas = {}
    for i, f in enumerate(("S", "H", "O")):
        p = tmp_path / f"{f}.fits"
        _write_channel(p, i + 10)
        metas[f] = SimpleNamespace(filename=p)

    # With stars (a black stars layer screens to a no-op, but exercises the path)
    stars_path = tmp_path / "stars.tif"
    write_tiff_image(stars_path, np.zeros((128, 128, 3), np.float32), {})
    stars = SimpleNamespace(filename=stars_path)
    out1 = tmp_path / "hubble.tif"
    pp._palette_image(str(out1), pp.palettes.hubble_vips,
                      metas["S"], metas["H"], metas["O"], stars, "Hubble")
    img = read_image(out1)
    assert img.shape == (128, 128, 3) and img.min() >= 0 and img.max() <= 1

    # Without stars (None) + the Foraxx formula
    out2 = tmp_path / "foraxx.tif"
    pp._palette_image(str(out2), pp.palettes.foraxx_vips,
                      metas["S"], metas["H"], metas["O"], None, "Foraxx")
    img2 = read_image(out2)
    assert img2.shape == (128, 128, 3) and np.all(np.isfinite(img2))


def test_bxt_narrow_keys_on_bxt_settings_only():
    """bxt_correct's cache key changes iff (mock, config.bxt) changes — the
    _StackKey guarantee applied to the color tail: tuning BXT re-derives BXT and
    its downstream, while an unrelated ProcessingConfig field never does."""
    from dataclasses import replace
    base = ProcessingConfig()
    args = ("m", "H", base, "chan")

    # Unrelated field (background_pedestal): identical projection → no re-derive.
    unrelated = replace(base, background_pedestal=base.background_pedestal + 0.1)
    assert pp._bxt_correct_narrow(args, {}) == \
        pp._bxt_correct_narrow(("m", "H", unrelated, "chan"), {})

    # A BXT knob: different projection → proper invalidation.
    sharper = replace(base, bxt=replace(base.bxt, sharpen_stars=0.5))
    assert pp._bxt_correct_narrow(args, {}) != \
        pp._bxt_correct_narrow(("m", "H", sharper, "chan"), {})

    # The mock flag flows into the key (mock path produces a different image).
    mocked = replace(base, development_settings=replace(
        base.development_settings, mock_pixinsight=True))
    assert pp._bxt_correct_narrow(args, {}) != \
        pp._bxt_correct_narrow(("m", "H", mocked, "chan"), {})


def test_sxt_narrow_keys_on_sxt_settings_only():
    """sxt_* keys on (mock, config.sxt); a BXT-only change must NOT touch it."""
    from dataclasses import replace
    base = ProcessingConfig()
    args = ("m", "O", base, "chan")

    bxt_only = replace(base, bxt=replace(base.bxt, sharpen_stars=0.5))
    assert pp._sxt_narrow(args, {}) == \
        pp._sxt_narrow(("m", "O", bxt_only, "chan"), {})

    changed = replace(base, sxt=replace(base.sxt, overlap=0.4))
    assert pp._sxt_narrow(args, {}) != \
        pp._sxt_narrow(("m", "O", changed, "chan"), {})
