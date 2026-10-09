"""Integration test for OSS colour post-processing.

Traces `color_post_process` on synthetic per-filter channels and runs it
through the executor (mock path — no rc-astro), asserting the LINEAR SCENE
FORMATS are produced as valid images (no palette is baked — every look is
assembled live in WGSL from these formats, finishes/stock.py) and that the
workflow_group wires the expected per-channel ops.
"""
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from attributes.astrometric_attributes import healpix_cells_from_poly
from context import context
from decisions.settings import DevelopmentSettings, ProcessingConfig, TilingSettings
from dependency_graph import run_graph
from dependency_graph.classes import Node
from dependency_graph.tracing import trace
from file_io.generic import read_image
from file_io.tiff_file_io import write_tiff_image
from metadata.classes import ImageMetadata
from mosaics.color import post_process as pp
from mosaics.types import Mosaic
from targets.types import Target
from units import ArcsecondsPerPixel, Degrees, Radians

_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_FOOTPRINT = SingleSphericalPolygon.from_radec(
    [98.0, 102.0, 102.0, 98.0], [-1, -1, 1, 1], center=[100.0, 0]
)
_HEALPIX = healpix_cells_from_poly(_FOOTPRINT)


def _write_channel(path: Path, seed: int, h: int, w: int) -> None:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.05, np.float32)
    img += 0.2 * np.exp(-(((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
                          / (2 * (min(h, w) / 4.0) ** 2)))
    for _ in range(max(15, (h * w) // 40000)):   # star count scales with area
        cy, cx = rng.uniform(8, h - 8), rng.uniform(8, w - 8)
        img += rng.uniform(0.2, 0.8) * np.exp(
            -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
        )
    write_tiff_image(path, np.clip(img, 0, 1).astype(np.float32), {"PIPELINE": "AstroFlow-OSS"})


def _mosaic() -> Mosaic:
    return Mosaic(
        name="color-test", content_hash="content_color", stable_hash="mosaic_color",
        target_ids=("a",), panel_target_ids=("a",),
        oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(0.0), Degrees(0.0)), image_scale=ArcsecondsPerPixel(8.0),
        rotation_angle=Radians(0.0), footprint=_FOOTPRINT, healpix_cells=_HEALPIX,
        projected_footprints=frozendict(),
    )


def _target(name: str, content_hash: str, footprint) -> Target:
    """A live panel Target (with a footprint) for `context.targets` — panels-as-
    tiles resolves `mosaic.targets` to project each panel's footprint bbox."""
    return Target(
        name=name, content_hash=content_hash, stable_hash=name,
        oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(100.0), Degrees(0.0)), image_scale=ArcsecondsPerPixel(8.0),
        rotation_angle=Radians(0.0), footprint=footprint,
        healpix_cells=healpix_cells_from_poly(footprint),
        projected_footprints=frozendict(), stack_ids=(name,))


def _count_by_op(graph) -> dict[str, int]:
    """Count nodes by op name, honoring PRESENTATION: a `{crop → op}`
    subgraph builder counts as its terminal op (`present_terminal` stamps
    `_op_name`), so the counts below read as processing-stage counts —
    the same numbers whether a stage is inline or scope-local. Interior
    nodes (the crops) are invisible by design."""
    counts: dict[str, int] = {}
    seen: set[int] = set()
    for k, deps in graph.items():
        for n in [k, *deps]:
            if isinstance(n, Node) and id(n) not in seen:
                seen.add(id(n))
                name = (getattr(n, "_op_name", None)
                        or getattr(n.task, "__name__", str(n.task)))
                counts[name] = counts.get(name, 0) + 1
    return counts


# EXHAUSTIVE (opt-in), and currently SLOW TO THE POINT OF NOT COMPLETING. This
# mock-PI (mock_pixinsight=True) end-to-end run traces a tiled color graph
# (~11n nodes at tile_px=512 on viewbox-sized channels) and drives it through
# run_graph. As of 2026-07-28 run_graph's scheduler is driver-bound on a graph
# this size: on a clean machine with 40G free it sat >4min at ~100% single-core
# CPU with every Ray worker IDLE and did NOT finish — a scheduler blowup, not
# memory pressure (the old "~131s isolated" note predates the tiling change).
# It hangs identically on main, so it's not a regression here. Excluded from the
# fast `make tests` suite (LIGHT = "not exhaustive …") and given a generous
# BOUNDED timeout — NOT timeout(0), which would hang forever — so a deliberate
# opt-in run self-terminates. Fixing run_graph's O(graph) scheduling is separate
# work. Run it deliberately:
#   PYTHONPATH=. pytest -m exhaustive mosaics/color/test_post_process.py
@pytest.mark.exhaustive
@pytest.mark.timeout(1200)
def test_color_post_process_traces_and_runs(test_context, tmp_path):
    # Size the synthetic channels to the mosaic viewbox — as the real merged
    # channels are — so the panel crops match their source. This is a single-panel
    # mosaic, so each rc-astro op fans into exactly one (crop_panel -> op ->
    # blend_panels); the multi-panel fan-out + incremental behaviour is proven in
    # test_panels_integration.py.
    mosaic = _mosaic()
    # Register the single panel so mosaic.targets resolves (panels-as-tiles
    # projects each panel's footprint); its footprint == the mosaic footprint.
    context.targets = {"a": _target("a", "target_a", _FOOTPRINT)}
    vb = mosaic.viewbox("Master")
    h, w = vb.num_pixels_y, vb.num_pixels_x
    # The group consumes per-filter LAYER mosaics (star removal is
    # pre-placement since 2026-09-02): starless carries scene + pedestal,
    # stars is a sparse residual. 90/10 split keeps the layer algebra
    # non-degenerate without rc-astro.
    starless, stars = {}, {}
    for i, f in enumerate(("S", "H", "O", "R", "G", "B")):
        p_sl = tmp_path / f"sl_{f}.tiff"
        _write_channel(p_sl, i, h, w)
        arr = read_image(p_sl).astype(np.float32)
        p_st = tmp_path / f"st_{f}.tiff"
        write_tiff_image(p_st, (arr * (1.0 / 9.0)).astype(np.float32),
                         {"PIPELINE": "AstroFlow-OSS"})
        # Construct the input metadata directly (compute_image_metadata is a
        # remote @node, callable only inside a graph); the ops only read
        # `.filename`, and `content_hash` keys the cache.
        def _meta(path, tag):
            return ImageMetadata(
                content_hash=tag, stable_hash=tag, filename=path,
                header={"FILTER": f}, width_pixels=w, height_pixels=h,
                channels=1, dtype=np.dtype("float32"),
            )
        starless[f] = _meta(p_sl, f"sl_{f}")
        stars[f] = _meta(p_st, f"st_{f}")

    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        tiling=TilingSettings(),
    )
    graph = trace(pp.color_post_process, mosaic, starless, stars, config)
    assert graph is not None

    counts = _count_by_op(graph)
    # Star work left the canvas entirely (native `bxt_channel`/`sxt_channel`
    # run pre-placement; SXT donuts on the resampled canvas, 2026-09-02).
    for gone in ("bxt_correct", "bxt_correct_only", "sxt_starless",
                 "sxt_stars_rgb", "combine_broadband_rgb", "sxt_stars",
                 "bxt_channel", "sxt_channel", "stars_channel",
                 "linear_narrowband", "autostretch_channel"):
        assert counts.get(gone) is None, f"canvas star op {gone} resurfaced"
    # NO baked palettes (2026-08-12): every look is assembled live in WGSL
    # from the linear scene formats. The formats themselves:
    for gone in ("combine_star_rgb", "hubble_image", "foraxx_image",
                 "hubble_starless", "foraxx_starless", "rgb_image",
                 "lrgb_image", "hargb_image", "halrgb_image"):
        assert counts.get(gone) is None, f"baked palette op {gone} resurfaced"
    assert counts.get("linear_narrowband_starless") is None  # full SHO set
    assert counts.get("linear_narrowband_stars") is None  # no consumer — never baked
    assert counts.get("linear_rgb_starless") == 1
    assert counts.get("linear_rgb_stars") == 1
    assert counts.get("linear_lum") is None       # no L channel in this fixture
    # The linear S/H/O viewer source: stack the 3 starless layer channels,
    # then NoiseXTerminator via process_in_panels (crop_panel + nxt per
    # panel), feathered into the deliverable. `linear_sho` is a traced
    # helper, not a node; NXT is the ONLY per-panel canvas op left.
    assert counts.get("linear_sho") is None
    assert counts.get("stack_sho_channels") == 1
    assert counts.get("blend_linear_sho") == 1        # the display="Linear SHO" deliverable (NXT's feather)
    assert counts.get("nxt") == 1                       # linear_sho NXT, per panel
    assert counts.get("blend_panels") is None     # NXT blends via blend_linear_sho
    # crop_panel is INTERIOR to the per-panel `{crop → op}` subgraphs
    # (delete_intermediates=True) — it never appears in the outer trace.
    assert counts.get("crop_panel") is None

    run_graph(graph)

    out = Path(context.output_dir)
    for op, name in (("linear_rgb_starless", "Linear RGB Starless"),
                     ("linear_rgb_stars", "Linear RGB Stars")):
        files = list(out.rglob(f"{op}/*.tif")) + list(out.rglob(f"{op}/*.tiff"))
        assert files, f"no {name} output produced under {op}/"
        img = read_image(files[0])
        assert img.ndim == 3 and img.shape[2] == 3, f"{name} not 3-channel: {img.shape}"
        assert np.all(np.isfinite(img))
        assert img.min() >= 0.0 and img.max() <= 1.0001

    # Linear SHO: a 3-band LINEAR BigTIFF master — no autostretch/palette.
    # In the mock path the tiled NXT + single-tile feather are identities,
    # so the stacked master must equal the starless S/H/O layers exactly
    # (proving no stretch; the WebGPU shader finishes them live).
    lin_files = (list(out.rglob("blend_linear_sho/*.tiff"))
                 + list(out.rglob("blend_linear_sho/*.tif")))
    assert lin_files, "no Linear SHO output produced under blend_linear_sho/"
    lin = read_image(lin_files[0]).astype(np.float32)
    assert lin.ndim == 3 and lin.shape[2] == 3, f"Linear SHO not 3-channel: {lin.shape}"
    assert np.all(np.isfinite(lin))
    for band, f in enumerate(("S", "H", "O")):
        raw = read_image(starless[f].filename).astype(np.float32)
        assert np.allclose(lin[:, :, band], raw, atol=1e-5), \
            f"Linear SHO band {f} differs from the starless layer (was it stretched?)"


def test_stars_channel_is_the_exact_residual(test_context, tmp_path):
    """`stars = channel − starless` clamped at 0, float32 — the identity the
    viewer's `starless + star_s·stars` reconstruction (and the lab's
    layer-linearity gate) relies on. Footprint zeros stay zero."""
    h, w = 64, 80
    rng = np.random.default_rng(7)
    channel = (0.005 + rng.random((h, w)) * 0.5).astype(np.float32)
    channel[:4] = 0.0                       # off-footprint border
    starless = np.where(channel > 0, 0.005 + 0.9 * (channel - 0.005),
                        np.float32(0)).astype(np.float32)
    p_ch, p_sl = tmp_path / "ch.tiff", tmp_path / "sl.tiff"
    write_tiff_image(p_ch, channel, {"FILTER": "H"})
    write_tiff_image(p_sl, starless, {"FILTER": "H"})

    def _meta(path, tag):
        return ImageMetadata(
            content_hash=tag, stable_hash=tag, filename=path, header={},
            width_pixels=w, height_pixels=h, channels=1,
            dtype=np.dtype("float32"))

    t = _target("panel", "panel_ch", _FOOTPRINT)
    # inspect.unwrap: the RAW body, below the decorator's metadata wrap.
    import inspect
    out = inspect.unwrap(pp.stars_channel._function)(
        t, "H", _meta(p_ch, "ch"), _meta(p_sl, "sl"))
    stars = read_image(Path(str(out))).astype(np.float32)
    if stars.ndim == 3:
        stars = stars[..., 0]
    assert stars.min() >= 0.0
    assert np.allclose(starless + stars, channel, atol=1e-6)
    assert np.all(stars[:4] == 0.0)         # zeros stay zeros


def test_bxt_narrow_keys_on_bxt_settings_only():
    """bxt_channel's cache key changes iff (mock, config.bxt) changes — the
    _StackKey guarantee applied to the native chain: tuning BXT re-derives BXT
    and its downstream, while an unrelated ProcessingConfig field never does."""
    from dataclasses import replace
    base = ProcessingConfig()
    t = SimpleNamespace(stable_hash="t")
    args = (t, "H", base, "chan")

    # Unrelated field (background_pedestal): identical projection → no re-derive.
    unrelated = replace(base, background_pedestal=base.background_pedestal + 0.1)
    assert pp._bxt_channel_narrow(args, {}) == \
        pp._bxt_channel_narrow((t, "H", unrelated, "chan"), {})

    # A BXT knob: different projection → proper invalidation.
    sharper = replace(base, bxt=replace(base.bxt, sharpen_stars=0.5))
    assert pp._bxt_channel_narrow(args, {}) != \
        pp._bxt_channel_narrow((t, "H", sharper, "chan"), {})

    # The mock flag flows into the key (mock path produces a different image).
    mocked = replace(base, development_settings=replace(
        base.development_settings, mock_pixinsight=True))
    assert pp._bxt_channel_narrow(args, {}) != \
        pp._bxt_channel_narrow((t, "H", mocked, "chan"), {})


def test_sxt_narrow_keys_on_sxt_settings_only():
    """sxt_channel keys on (mock, config.sxt); a BXT-only change must NOT
    touch it (the BXT dependence rides in through `channel`, its input)."""
    from dataclasses import replace
    base = ProcessingConfig()
    t = SimpleNamespace(stable_hash="t")
    args = (t, "O", base, "chan")

    bxt_only = replace(base, bxt=replace(base.bxt, sharpen_stars=0.5))
    assert pp._sxt_channel_narrow(args, {}) == \
        pp._sxt_channel_narrow((t, "O", bxt_only, "chan"), {})

    changed = replace(base, sxt=replace(base.sxt, overlap=0.4))
    assert pp._sxt_channel_narrow(args, {}) != \
        pp._sxt_channel_narrow((t, "O", changed, "chan"), {})



def test_mono_layer_outputs_say_which_layer_they_are(tmp_path, monkeypatch):
    """`_save_mono_vips` writes the starless/stars mono deliverables, and its
    header carried only FILTER + PIPELINE — so a stars mosaic and its starless
    twin were indistinguishable outside the op's output directory. Both halves
    are pinned: the writer stamps LAYER, and each caller declares its own."""
    import inspect

    import pyvips
    from file_io.tiff_file_io import read_tiff_header

    out = tmp_path / "mono.tiff"
    img = pyvips.Image.black(64, 48).cast("float") + 1.0
    pp._save_mono_vips(str(out), img, "Ha", layer="stars")
    assert read_tiff_header(out)["LAYER"] == "stars"

    pp._save_mono_vips(str(out), img, "Ha")
    assert "LAYER" not in read_tiff_header(out)

    seen = []
    monkeypatch.setattr(pp, "_open_mono_vips", lambda meta: img)
    monkeypatch.setattr(pp, "_save_mono_vips",
                        lambda path, i, f, layer=None: seen.append(layer))
    for op in (pp.linear_lum, pp.linear_lum_stars):
        # filename_fn resolves the output dir through context; the layer
        # literal is what is under test, so stub the naming out
        monkeypatch.setattr(op, "filename_fn", lambda *a: str(out))
        # past beartype — these bodies only pick a filename and forward
        inspect.unwrap(op._function)(None, None)
    monkeypatch.setattr(pp.stars_channel, "filename_fn", lambda *a: str(out))
    inspect.unwrap(pp.stars_channel._function)(None, "Ha", None, None)
    assert seen == ["starless", "stars", "stars"]
