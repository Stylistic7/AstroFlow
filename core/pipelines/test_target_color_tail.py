"""The Target colour tail — a Target IS a single-panel mosaic.

`for_target` (Hybrid: PI Master + OSS Mosaic) runs the exact mosaic machinery
on a Target: each channel is placed into the target's own viewbox and merged
through the self-cleaning `{place → merge}` scope, then `color_post_process`
produces the same colour deliverables mosaics get. These tests pin the three
load-bearing pieces WITHOUT running the graph:

  * the single-panel-mosaic contract on `Target` (`targets`/`strips`/`viewbox`);
  * every tail op's `mosaic` annotation ACCEPTS a Target at runtime — the
    `MosaicLike`-imported-as-`Mosaic` rebinding. beartype enforces annotations
    on every `@node` (cached_decorator applies it), and the annotations
    themselves cannot be edited: they serialize into the structural op hash,
    so widening them in place would re-key every existing mosaic op;
  * the traced tail on a Target: one presented `{place → merge}` builder per
    filter, the colour deliverables, and every tap routed to the TARGET.
"""
import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from frozendict import frozendict
from spherical_geometry.polygon import SingleSphericalPolygon  # type: ignore

from attributes.astrometric_attributes import healpix_cells_from_poly
from decisions.settings import DevelopmentSettings, ProcessingConfig, TilingSettings
from dependency_graph.tracing import trace
from metadata.classes import ImageMetadata
from targets.types import Target
from units import ArcsecondsPerPixel, Degrees, Radians

_DATE = datetime.datetime(2024, 2, 5, 16, 29, 43)
_FOOTPRINT = SingleSphericalPolygon.from_radec(
    [98.0, 102.0, 102.0, 98.0], [-1, -1, 1, 1], center=[100.0, 0]
)
_FILTERS = ["B", "G", "H", "O", "R", "S"]


def _target() -> Target:
    return Target(
        name="tail-test", content_hash="target_content", stable_hash="target_tail",
        oldest_date=_DATE, newest_date=_DATE,
        center=(Degrees(100.0), Degrees(0.0)), image_scale=ArcsecondsPerPixel(8.0),
        rotation_angle=Radians(0.0), footprint=_FOOTPRINT,
        healpix_cells=healpix_cells_from_poly(_FOOTPRINT),
        projected_footprints=frozendict(), stack_ids=("s0",))


def test_target_single_panel_mosaic_contract():
    """`targets`/`strips`/`viewbox` make a Target its own one-panel mosaic."""
    from mosaics.viewbox import compute_viewbox
    t = _target()
    assert t.targets == (t,)
    assert t.strips == ((t.id,),)
    # Viewbox equality via content_hash: `projection` has no __eq__.
    assert t.viewbox("Master").content_hash == \
        compute_viewbox("Master", t).content_hash


def test_tail_ops_accept_target():
    """Every keyed tail op's `mosaic` annotation must accept a Target — the
    import rebinding in each tail module is what widens it; reverting any one
    of those imports makes beartype reject the Target on the worker."""
    from beartype.door import is_bearable
    import mosaics.color.post_process as pp
    import mosaics.merging.oss as oss
    import mosaics.panels as panels
    import mosaics.placement.astroflow as placement

    t = _target()
    for mod, name in [
        (panels, "crop_panel"), (panels, "blend_panels"),
        (panels, "combine_panels"),
        (oss, "merge_images_oss"),
        (placement, "place_image"),
        (pp, "linear_narrowband"), (pp, "linear_rgb_stars"),
        (pp, "blend_linear_sho"),
    ]:
        fn = getattr(mod, name)
        for attr in ("_function", "__wrapped__", "func"):
            fn = getattr(fn, attr, fn)
        ann = fn.__annotations__.get("mosaic")
        assert ann is not None, f"{name}: mosaic param lost its annotation"
        assert is_bearable(t, ann), f"{name}: beartype would reject a Target"


def test_target_tail_traces_with_target_taps(tmp_path, monkeypatch):
    """Trace `for_target`'s composition (fake channel sinks in place of
    `channels_for_target` — the stack backbone is PI Master's, proven
    elsewhere): one presented merge per filter, the colour deliverables
    present, and EVERY tap routed to the Target."""
    import targets.target_channel as tc
    monkeypatch.setattr(tc, "filters_for_target", lambda t: list(_FILTERS))

    t = _target()
    channels = {}
    for i, f in enumerate(_FILTERS):
        p = tmp_path / f"chan_{f}.tiff"
        from file_io.tiff_file_io import write_tiff_image
        write_tiff_image(p, np.zeros((8, 8), np.float32), {"FILTER": f})
        channels[(t.id, f)] = ImageMetadata(
            content_hash=f"chan_{f}", stable_hash=f"chan_{f}", filename=p,
            header={"FILTER": f}, width_pixels=8, height_pixels=8,
            channels=1, dtype=np.dtype("float32"),
        )

    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        tiling=TilingSettings(),
    )

    def _tail(target, channels, config):
        from mosaics.color.post_process import color_post_process
        from pipelines.hybrid_pi_oss_mosaic import (
            _measured_viewbox, _merged_channels,
        )
        # Trace-level only: the viewbox is a recorded HANDLE here (the node
        # body never runs, so the synthetic channels' absent plate solutions
        # are never read) — exactly the state the real pipeline traces in.
        viewbox = _measured_viewbox(channels, single_panel=True)
        merged = _merged_channels(target, {target.id: target}, channels,
                                  config, viewbox)
        color = color_post_process(target, merged, config, viewbox)
        return {**merged, **color}

    graph = trace(_tail, t, channels, config)
    assert graph is not None

    # One presented `{N× place → merge}` builder per filter (builder ≡
    # terminal: `_op_name` is stamped by present_terminal).
    merges = [n for n in graph
              if getattr(n, "_op_name", None) == "merge_images_oss"]
    assert len(merges) == len(_FILTERS)

    from compute_graph.compute_graph import extract_pipeline_taps
    taps = list(extract_pipeline_taps(graph))
    assert taps, "no taps extracted from the target tail"
    routed = {(object_type, object_id) for _, (object_type, object_id, _, _) in taps}
    assert routed == {("Target", t.stable_hash)}, (
        f"taps must route to the TARGET, got {routed}")
    displays = {display for _, (_, _, display, _) in taps}
    # The colour output is the LINEAR SCENE FORMATS only — no baked palettes
    # (2026-08-12; looks are assembled live in WGSL, finishes/stock.py).
    assert {"Mosaic", "Linear SHO", "Linear Narrowband", "Linear RGB",
            "Linear RGB Stars"} <= displays, (
        f"scene formats missing from taps: {displays}")
    assert not ({"Hubble", "Foraxx", "Hubble Starless", "Foraxx Starless",
                 "Stars", "RGB", "LRGB", "HaRGB", "HaLRGB"} & displays), (
        f"baked palette tap resurfaced: {displays}")
    # Full SHO set → no partial-set starless variant; no Lum → no L tap.
    assert not ({"Linear Narrowband Starless", "Linear L"} & displays), displays


def test_lrgb_target_traces_rgb_and_lrgb(tmp_path, monkeypatch):
    """A broadband LRGB target (real-world filter aliases) produces the RGB
    and LRGB deliverables and none of the narrowband palettes."""
    import targets.target_channel as tc
    filters = ["Red", "Green", "Blue", "Lum"]
    monkeypatch.setattr(tc, "filters_for_target", lambda t: list(filters))

    t = _target()
    channels = {}
    for i, f in enumerate(filters):
        p = tmp_path / f"chan_{f}.tiff"
        from file_io.tiff_file_io import write_tiff_image
        write_tiff_image(p, np.zeros((8, 8), np.float32), {"FILTER": f})
        channels[(t.id, f)] = ImageMetadata(
            content_hash=f"chan_{f}", stable_hash=f"chan_{f}", filename=p,
            header={"FILTER": f}, width_pixels=8, height_pixels=8,
            channels=1, dtype=np.dtype("float32"),
        )

    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        tiling=TilingSettings(),
    )

    def _tail(target, channels, config):
        from mosaics.color.post_process import color_post_process
        from pipelines.hybrid_pi_oss_mosaic import (
            _measured_viewbox, _merged_channels,
        )
        # Trace-level only: the viewbox is a recorded HANDLE here (the node
        # body never runs, so the synthetic channels' absent plate solutions
        # are never read) — exactly the state the real pipeline traces in.
        viewbox = _measured_viewbox(channels, single_panel=True)
        merged = _merged_channels(target, {target.id: target}, channels,
                                  config, viewbox)
        color = color_post_process(target, merged, config, viewbox)
        return {**merged, **color}

    graph = trace(_tail, t, channels, config)

    from compute_graph.compute_graph import extract_pipeline_taps
    displays = {d for _, (_, _, d, _) in extract_pipeline_taps(graph)}
    assert {"Mosaic", "Linear RGB", "Linear RGB Stars", "Linear L"} <= displays, (
        f"broadband scene formats missing: {displays}")
    # No baked palettes; no narrowband channel → no narrowband formats.
    assert not ({"Stars", "RGB", "LRGB", "HaRGB", "HaLRGB", "Hubble",
                 "Foraxx", "Linear SHO", "Linear Narrowband",
                 "Linear Narrowband Starless"} & displays), displays


def test_galaxy_target_traces_ha_blend_taps(tmp_path, monkeypatch):
    """The galaxy-with-Hα case (LRGB + Ha): BOTH Hα looks appear as separate
    taps — HaRGB (Hα in chrominance: starless Hα added to linear R) and
    HaLRGB (Hα in lightness: L lightened by starless Hα, colour from the
    plain RGB) — alongside the plain RGB/LRGB for comparison."""
    import targets.target_channel as tc
    filters = ["Red", "Green", "Blue", "Lum", "Ha"]
    monkeypatch.setattr(tc, "filters_for_target", lambda t: list(filters))

    t = _target()
    channels = {}
    for i, f in enumerate(filters):
        p = tmp_path / f"chan_{f}.tiff"
        from file_io.tiff_file_io import write_tiff_image
        write_tiff_image(p, np.zeros((8, 8), np.float32), {"FILTER": f})
        channels[(t.id, f)] = ImageMetadata(
            content_hash=f"chan_{f}", stable_hash=f"chan_{f}", filename=p,
            header={"FILTER": f}, width_pixels=8, height_pixels=8,
            channels=1, dtype=np.dtype("float32"),
        )

    config = ProcessingConfig(
        development_settings=DevelopmentSettings(mock_pixinsight=True),
        tiling=TilingSettings(),
    )

    def _tail(target, channels, config):
        from mosaics.color.post_process import color_post_process
        from pipelines.hybrid_pi_oss_mosaic import (
            _measured_viewbox, _merged_channels,
        )
        # Trace-level only: the viewbox is a recorded HANDLE here (the node
        # body never runs, so the synthetic channels' absent plate solutions
        # are never read) — exactly the state the real pipeline traces in.
        viewbox = _measured_viewbox(channels, single_panel=True)
        merged = _merged_channels(target, {target.id: target}, channels,
                                  config, viewbox)
        color = color_post_process(target, merged, config, viewbox)
        return {**merged, **color}

    graph = trace(_tail, t, channels, config)

    from compute_graph.compute_graph import extract_pipeline_taps
    displays = {d for _, (_, _, d, _) in extract_pipeline_taps(graph)}
    # Scene formats for the galaxy case: raw Hα (with stars — HII knots
    # intact), the PARTIAL starless variant, the RGB pair, and L. The
    # HaLRGB/LRGB looks assemble live from exactly these.
    assert {"Linear Narrowband", "Linear Narrowband Starless",
            "Linear RGB", "Linear RGB Stars", "Linear L",
            "Mosaic"} <= displays, displays
    assert not ({"Stars", "RGB", "LRGB", "HaRGB", "HaLRGB", "Hubble",
                 "Foraxx", "Linear SHO"} & displays), (
        f"baked palette tap resurfaced: {displays}")
